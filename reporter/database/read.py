import itertools
from datetime import datetime, timedelta
from decimal import Decimal
from logging import Logger
from typing import Any, Dict, List, Tuple
from xml.etree.ElementTree import fromstring

from sqlalchemy import Date, Integer, cast, extract, func
from sqlalchemy.orm import Session
from tqdm import tqdm

from reporter.core.operation import find_operation
from reporter.database.misc import in_jst, in_utc
from reporter.database.model import Summary, Intake, IntakeSeq
from reporter.util.constant import UTC, Code, Phase, SeqType
from reporter.util.conversion import stringify_ric_seqtype


class Alignment:

    def __init__(self,
                 summary_id: str,
                 summary: str,
                 date_start: str,
                 date_end: str,
                 processed_tokens: List[str],
                 chart: Dict[str, List[str]]):

        self.summary_id = summary_id
        self.summary = summary
        self.date_start = date_start
        self.date_end = date_end
        self.processed_tokens = processed_tokens
        self.chart = chart

    def to_dict(self) -> Dict[str, Any]:
        return {'summary_id': self.summary_id,
                'summary': self.summary,
                'date_start': self.date_start,
                'date_end': self.date_end,
                'processed_tokens': self.processed_tokens,
                **self.chart}


def are_summaries_ready(session: Session):

    return session \
        .query(Summary) \
        .filter(Summary.summary.isnot(None)) \
        .first() is not None


def fetch_rics(session: Session) -> List[str]:
    results = session.query(Intake.ric).distinct()
    return [result.ric for result in results]


def fetch_date_range(session: Session) -> Tuple[datetime, datetime]:
    results = session.query(func.min(Intake.t), func.max(Intake.t)).first()
    return results


def fetch_prices_of_a_day(session: Session,
                          ric: str,
                          jst: datetime) -> List[Tuple[datetime, Decimal]]:
    results = session \
        .query(func.to_char(in_utc(Intake.t), 'YYYY-MM-DD HH24:MI:SS').label('t'),
               Intake.val) \
        .filter(cast(in_jst(Intake.t), Date) == jst.date(), Intake.ric == ric) \
        .order_by(Intake.t) \
        .all()

    return [(datetime.strptime(r.t, '%Y-%m-%d %H:%M:%S').replace(tzinfo=UTC), r.val)
            for r in results]


def fetch_max_t_of_prev_trading_day(session: Session, ric: str, t: datetime) -> int:

    assert(t.tzinfo == UTC)

    prev_day = t + timedelta(hours=9) - timedelta(days=1)

    return session \
        .query(extract('epoch', func.max(Intake.t))) \
        .filter(cast(in_jst(Intake.t), Date) <= prev_day.date(), Intake.ric == ric) \
        .scalar()


def fetch_latest_vals(session: Session,
                      t_start: datetime,
                      t_end: datetime,
                      ric: str,
                      seqtype: SeqType,
                      intake_seqs: List) -> Tuple[str, List[str]]:

    #min_t = t - timedelta(days=7)
    t_list = []
    r = []
    for key in intake_seqs.keys():
        for seq in intake_seqs[key]:
            import pytz
            utc = pytz.UTC
            if seq['ric'] == ric and seq['seqtype'] == seqtype.value \
               and utc.localize(seq['date_consumed']) <= t_end \
               and utc.localize(seq['date_consumed']) >= t_start:
                t_list.append(seq)
                
    #for item in t_list:
        #print(utc.localize(seq['date_consumed']),t)
        #if utc.localize(seq['date_consumed']) == t:
            #r.append(item)
            
    #input(r)
    for item in t_list:
        r.append(item['vals'][0])
    #if len(r) > 1:
        #r = None
    #t = session \
        #.query(func.max(IntakeSeq.t)) \
        #.filter(IntakeSeq.ric == ric,
                #IntakeSeq.seqtype == seqtype.value,
                #IntakeSeq.t <= t,
                #IntakeSeq.t > min_t) \
        #.scalar()
    #r = session \
        #.query(IntakeSeq) \
        #.filter(IntakeSeq.ric == ric,
                #IntakeSeq.seqtype == seqtype.value,
                #IntakeSeq.t == t) \
        #.one_or_none()
    #input(t_list)
    #input(t)
    #input(r)
    #input(r)
    #input([stringify_ric_seqtype(ric, seqtype),
            #[] if r is None else ['{:.2f}'.format(v) for v in r]])
    #input(r)
    return (stringify_ric_seqtype(ric, seqtype),
            [] if r is None else ['{:.2f}'.format(v) for v in r])


def load_alignments_from_db(session: Session, phase: Phase, logger: Logger, intakes: List, intake_seqs: List, summaries: List) -> List[Alignment]:

    #headlines = session \
        #.query(Summary.summary_id,
               #Summary.date_start,
               #cast(extract('epoch', Summary.date_start), Integer).label('unixtime')) \
        #.filter(Summary.phase == phase.value) \
        #.order_by(Summary.date_start) \
        #.all()
    #headlines = list(headlines)

    rics = ['MFP']

    alignments = []
    seqtypes = [SeqType.RawShort, SeqType.RawLong,
                SeqType.MovRefShort, SeqType.MovRefLong,
                SeqType.NormMovRefShort, SeqType.NormMovRefLong,
                SeqType.StdShort, SeqType.StdLong]
    logger.info('start creating alignments between summaries and intake sequences.')
    
    for h in tqdm(summaries):
        #input(h)

        # Find the latest prices before the article is published
        chart = dict([fetch_latest_vals(session, h['date_start'], h['date_end'], ric, seqtype, intake_seqs)
                      for (ric, seqtype) in itertools.product(rics, seqtypes)])
        #print(rics, seqtypes)
        # Replace tags with price tags
        tag_tokens = []
        #input(Code.N225.value)
        
        short_term_vals = chart[stringify_ric_seqtype(Code.MFP.value, SeqType.RawShort)]
        long_term_vals = chart[stringify_ric_seqtype(Code.MFP.value, SeqType.RawLong)]

        processed_tokens = []
        for i in range(len(tag_tokens)):
            t = tag_tokens[i]
            if t.startswith('<yen val="') and t.endswith('"/>'):
                ref = fromstring(t).attrib['val']

                if len(short_term_vals) > 0 and len(long_term_vals) > 0:

                    prev_trading_day_close = Decimal(long_term_vals[0])
                    latest = Decimal(short_term_vals[0])
                    p = find_operation(ref, prev_trading_day_close, latest)
                    processed_tokens.append(p)
                else:
                    processed_tokens.append('<yen val="z"/>')
            else:
                processed_tokens.append(tag_tokens[i])
        alignment = Alignment(h['summary_id'], h, str(h['date_start']), str(h['date_end']), processed_tokens, chart)
        alignments.append(alignment.to_dict())
    logger.info('end creating alignments between summaries and intake sequences.')
    return alignments
