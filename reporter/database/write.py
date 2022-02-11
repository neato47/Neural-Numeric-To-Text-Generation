import csv
import gzip
from datetime import datetime
from decimal import Decimal
from logging import Logger
from math import isclose
from pathlib import Path
from typing import List

import pandas
from janome.tokenizer import Tokenizer
from pytz import UTC
from sqlalchemy.orm.session import Session
from sqlalchemy.sql.expression import extract
from tqdm import tqdm

from reporter.database.misc import in_jst
from reporter.database.model import (
    Close,
    Summary,
    Instrument,
    Intake,
    IntakeSeq
)
from reporter.preprocessing.text import (
    is_interesting,
    kansuuzi2number,
    replace_prices_with_tags
)
from reporter.resource.reuters import ric2filename
from reporter.util.config import Span
from reporter.util.constant import (
    DOMESTIC_INDEX,
    EQUITY,
    FUTURES,
    JST,
    N_LONG_TERM,
    N_SHORT_TERM,
    NIKKEI_DATETIME_FORMAT,
    REUTERS_DATETIME_FORMAT,
    Code,
    Phase,
    SeqType
)
from reporter.util.exchange import ClosingTime


def get_data(session: Session,
                  dir_intakes: Path,
                  missing_rics: List[str],
                  dir_resources: Path,
                  logger: Logger) -> None:

    #ct = ClosingTime(dir_resources)
    #insert_instruments(session, dir_resources / Path('ric.csv'), logger)
    seqtypes = [SeqType.RawShort, SeqType.RawLong,
                SeqType.MovRefShort, SeqType.MovRefLong,
                SeqType.NormMovRefShort, SeqType.NormMovRefLong,
                SeqType.StdShort, SeqType.StdLong]

    for ric in missing_rics:

        filename = ric2filename(dir_intakes, ric, extension='csv')

        intake_seqs = dict((seqtype, []) for seqtype in seqtypes)
        
        with open(filename, mode='rt') as f:
            dataframe = pandas.read_table(f, delimiter=',')
            column = 'Calories' #if int(dataframe[['Last']].dropna().count()) == 0 else 'Last'
            mean = float(dataframe[[column]].mean())
            std = float(dataframe[[column]].std())

            f.seek(0)
            N = sum(1 for _ in f) - 1

            f.seek(0)
            reader = csv.reader(f, delimiter=',')
            next(reader)
            fields = next(reader)
            ric = fields[0]

            #stock_exchange = session \
                #.query(Instrument.exchange) \
                #.filter(Instrument.ric == ric) \
                #.scalar()
            #if stock_exchange is None:
                #stock_exchange = 'TSE'
            #get_close_utc = ct.func_get_close_t(stock_exchange)

            logger.info('start importing {}'.format(f.name))

            f.seek(0)
            column_names = next(reader)
            # Some indices contain an additional column
            shift = 1 if column_names[1] == 'Alias Underlying RIC' else 0

            intakes = []
            intakes_ = []
            raw_short_vals = []
            raw_long_vals = []
            raw_mov_ref_long_vals = []
            raw_mov_ref_short_vals = []
            std_long_vals = []
            std_short_vals = []
            prev_row_t_start = None
            prev_row_t_end = None
            max_mov_ref_long_val = float('-inf')
            min_mov_ref_long_val = float('inf')
            max_mov_ref_short_val = float('-inf')
            min_mov_ref_short_val = float('inf')

            for i in tqdm(range(N)):
                fields = next(reader)
                ric = "MFP"
                date_consumed = fields[0]
                calories = fields[1]
                
                from time import localtime, gmtime
                import calendar
                utc_offset = calendar.timegm(gmtime()) - calendar.timegm(localtime())
                #date_consumed = fields[2 + shift].replace('Z', '+0000')
                #utc_offset = int(fields[3 + shift])
                #if ric == Code.SPX.value:
                    #utc_offset += 1
                #last = fields[8 + shift].strip()
                #close_bid = fields[14 + shift].strip()

                #if last == '' and close_bid == '':
                    #continue
                val = Decimal(calories)
                std_val = (float(val) - mean) / std
                try:
                    date_consumed = datetime.strptime(date_consumed, "%m/%d/%Y")
                except ValueError:
                    logger.info('ValueError: {}, {}, {}'.format(ric, date_consumed, val))
                    continue

                #if prev_row_t_start is not None and prev_row_t_end is not None:

                    #if prev_row_t_start == t_start or prev_row_t_end == t_end:
                        #continue

                    ##close_time = get_close_utc(utc_offset)
                    
                    #start_datetime = datetime(t_start.year, t_start.month, t_start.day)
                    #end_datetime = datetime(t_end.year, t_end.month, t_end.day)

                    #if prev_row_t_start < start_datetime: #and start_datetime <= date_consumed:
                        #intakes_.append(Close(ric, date_consumed).to_dict())

                        #if len(raw_long_vals) > 1:
                            #raw_mov_ref_long_val = float(val) - raw_long_vals[0]
                            #raw_mov_ref_long_vals = [raw_mov_ref_long_val] + raw_mov_ref_long_vals \
                                #if len(raw_mov_ref_long_vals) < N_LONG_TERM \
                                #else [raw_mov_ref_long_val] + raw_mov_ref_long_vals[:-1]
                            #price_seqs[SeqType.MovRefLong] \
                                #.append(PriceSeq(ric, SeqType.MovRefLong, date_consumed, raw_mov_ref_long_vals).to_dict())
                            #max_mov_ref_long_val = raw_mov_ref_long_val \
                                #if raw_mov_ref_long_val > max_mov_ref_long_val \
                                #else max_mov_ref_long_val
                            #min_mov_ref_long_val = raw_mov_ref_long_val \
                                #if raw_mov_ref_long_val < min_mov_ref_long_val \
                                #else min_mov_ref_long_val

                        #raw_long_vals = [float(val)] + raw_long_vals \
                            #if len(raw_long_vals) < N_LONG_TERM \
                            #else [float(val)] + raw_long_vals[:-1]
                        #price_seqs[SeqType.RawLong] \
                            #.append(PriceSeq(ric, SeqType.RawLong, date_consumed, raw_long_vals).to_dict())

                        #std_long_vals = [std_val] + std_long_vals \
                            #if len(std_long_vals) < N_LONG_TERM \
                            #else [std_val] + std_long_vals[:-1]
                        #price_seqs[SeqType.StdLong] \
                            #.append(PriceSeq(ric, SeqType.StdLong, date_consumed, std_long_vals).to_dict())

                intakes.append(Intake(ric, date_consumed, utc_offset, val).to_dict())

                if len(raw_short_vals) > 1 and len(raw_long_vals) > 2:
                    raw_mov_ref_short_val = float(val) - raw_long_vals[1 if date_consumed == close_datetime else 0]
                    raw_mov_ref_short_vals = [raw_mov_ref_short_val] + raw_mov_ref_short_vals \
                        if len(raw_mov_ref_short_vals) < N_SHORT_TERM \
                        else [raw_mov_ref_short_val] + raw_mov_ref_short_vals[:-1]
                    intake_seqs[SeqType.MovRefShort] \
                        .append(IntakeSeq(ric, SeqType.MovRefShort, date_consumed, raw_mov_ref_short_vals).to_dict())
                    max_mov_ref_short_val = raw_mov_ref_short_val \
                        if raw_mov_ref_short_val > max_mov_ref_short_val \
                        else max_mov_ref_short_val
                    min_mov_ref_short_val = raw_mov_ref_short_val \
                        if raw_mov_ref_short_val < min_mov_ref_short_val \
                        else min_mov_ref_short_val

                raw_short_vals = [float(val)] + raw_short_vals \
                    if len(raw_short_vals) < N_SHORT_TERM \
                    else [float(val)] + raw_short_vals[:-1]
                intake_seqs[SeqType.RawShort] \
                    .append(IntakeSeq(ric, SeqType.RawShort, date_consumed, raw_short_vals).to_dict())

                std_short_vals = [std_val] + std_short_vals \
                    if len(std_short_vals) < N_SHORT_TERM \
                    else [std_val] + std_short_vals[:-1]
                intake_seqs[SeqType.StdShort] \
                    .append(IntakeSeq(ric, SeqType.StdShort, date_consumed, std_short_vals).to_dict())
                #prev_row_t = date_consumed
            
            try:
                session.execute(Intake.__table__.insert(), intakes)
                #from zope.sqlalchemy import mark_changed
                #mark_changed(session)
                #session.execute(Close.__table__.insert(), close_prices)
            except:
                pass
            
            for seqtype in seqtypes:
                if seqtype == SeqType.NormMovRefShort:
                    intake_seqs[seqtype] = \
                        [IntakeSeq(ric, SeqType.NomMovRefShort, p['date_consumed'], None)
                         for p in intake_seqs[SeqType.MovRefShort]] \
                        if isclose(max_mov_ref_long_val, min_mov_ref_short_val) \
                        else [IntakeSeq(ric, SeqType.NormMovRefShort, p['date_consumed'],
                                       [(2 * v - (max_mov_ref_short_val + min_mov_ref_short_val)) /
                                        (max_mov_ref_short_val - min_mov_ref_short_val)
                                        for v in p['vals']]).to_dict()
                              for p in intake_seqs[SeqType.MovRefShort]]
                elif seqtype == SeqType.NormMovRefLong:
                    intake_seqs[seqtype] = \
                        [IntakeSeq(ric, SeqType.NomMovRefLong, p['date_consumed'], None)
                         for p in intake_seqs[SeqType.MovRefLong]] \
                        if isclose(max_mov_ref_long_val, min_mov_ref_long_val) \
                        else [IntakeSeq(ric, SeqType.NormMovRefLong, p['date_consumed'],
                                       [(2 * v - (max_mov_ref_long_val + min_mov_ref_long_val)) /
                                        (max_mov_ref_long_val - min_mov_ref_long_val)
                                        for v in p['vals']]).to_dict()
                              for p in intake_seqs[SeqType.MovRefLong]]
                try:
                    session.execute(IntakeSeq.__table__.insert(), intake_seqs[seqtype])
                except:
                    pass
                
            session.commit()

            logger.info('end importing {}'.format(ric))

    return intakes, intake_seqs
def get_summaries(session: Session,
                     dir_summaries: Path,
                     train_span: Span,
                     valid_span: Span,
                     test_span: Span,
                     logger: Logger) -> None:

    dests = list(dir_summaries.glob('*.csv.gz')) + list(dir_summaries.glob('*.csv'))
    for dest in dests:
        with gzip.open(str(dest), mode='rt') if dest.suffix == '.gz' else dest.open(mode='r') as f:

            N = sum(1 for _ in f) - 1
            f.seek(0)
            reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_ALL)
            next(reader)
            fields = next(reader)
            #date_start = fields[0]
            #date_end = fields[1]
            #input(fields)
            #if 'Z' not in date_consumed or '+' not in date_consumed:
                #date_consumed = date_consumed + '+0000'
            #date_start = datetime.strptime(date_start, "%m/%d/%Y")
            #date_end = datetime.strptime(date_end, "%m/%d/%Y")
            #first = session \
                #.query(Headline) \
                #.filter(extract('year', in_jst(Headline.date_consumed)) == date_consumed.year) \
                #.first()
            #if first is not None:
                #return

            logger.info('start {}'.format(f.name))

            f.seek(0)
            next(reader)
            summaries = []
            for _ in tqdm(range(N)):
                fields = next(reader)
                #date_consumed = fields[1]
                #if 'Z' not in date_consumed or '+' not in date_consumed:
                    #date_consumed = date_consumed + '+0000'
                import pytz
                timezone = pytz.timezone("America/New_York")
                date_start = fields[1]
                date_end = fields[2]      
                date_start = datetime.strptime(date_start, "%m/%d/%Y")
                date_end = datetime.strptime(date_end, "%m/%d/%Y")
                
                date_start = timezone.localize(date_start)
                date_end = timezone.localize(date_end)
                
                summary_id = fields[0]
                summary = fields[3]
                isins = None #if fields[25] == '' else fields[25].split(':')
                countries = None #if fields[36] == '' else fields[36].split(':')
                categories = None #if fields[37] == '' else fields[37].split(':')
                keywords_headline = None #if fields[-2] == '' else fields[-2].split(':')
                keywords_article = None #if fields[-1] == '' else fields[-1].split(':')
                #try:
                    #date_consumed = datetime.strptime(date_consumed, NIKKEI_DATETIME_FORMAT)
                #except ValueError:
                    #message = 'ValueError: {}, {}, {}'
                    #logger.info(message.format(date_consumed, article_id, headline))
                    #continue
                    
                if train_span.start <= date_start and date_end < train_span.end:
                    phase = Phase.Train.value
                elif valid_span.start <= date_start and date_end < valid_span.end:
                    phase = Phase.Valid.value
                elif test_span.start <= date_start and date_end < test_span.end:
                    phase = Phase.Test.value
                else:
                    phase = None

                summaries.append({'summary_id': summary_id,
                                  'date_start': date_start,
                                  'date_end': date_end,
                                  'summary': summary,
                                  'phase': phase})
            try:
                session.execute(Summary.__table__.insert(), summaries)
            except:
                pass
            session.commit()
    return summaries

def update_summaries(session: Session, user_dict: Path, logger: Logger) -> None:

    query_result = session \
        .query(Summary) \
        .all()
    summaries = list(query_result)

    if len(summaries) == 0:
        return

    tokenizer = Tokenizer(str(user_dict))
    mappings = []

    logger.info('start updating summaries')
    for summary in tqdm(summaries):
        h = summary.summary
        #is_about_di = headline.categories is not None and \
            #DOMESTIC_INDEX in headline.categories

        # We stopped using `is_template` because the size of the dataset decreased and the result got worse.
        # if is_template(h) or not is_interesting(h) or not is_about_di:
        if not is_interesting(h):
            mappings.append({
                'summary_id': summary.summary_id,
                'is_used': False
            })
            continue

        tokens = kansuuzi2number([token.surface
                                  for token in tokenizer.tokenize(h)])
        tag_tokens = replace_prices_with_tags(tokens)

        mappings.append({
            'summary_id': summary.summary_id,
            'simple_summary': h,
            'tokens': tokens,
            'tag_tokens': tag_tokens,
            'is_used': True,
        })
    session.bulk_update_mappings(Summary, mappings)
    session.commit()
    logger.info('end updating summaries')


def insert_instruments(session: Session, dest_ric: Path, logger: Logger) -> None:
    with dest_ric.open(mode='r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)
        for fields in reader:
            ric = fields[0]
            desc = fields[1]
            currency = fields[2]
            type_ = fields[3]
            exchange = fields[4] if type_ in [EQUITY, FUTURES] else None
            instrument = Instrument(ric, desc, currency, type_, exchange)
            session.merge(instrument)
    session.commit()
