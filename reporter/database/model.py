from datetime import datetime
from typing import Any, Dict, List, Union

from sqlalchemy import (
    TIMESTAMP,
    Boolean,
    Column,
    Float,
    Integer,
    Numeric,
    String,
    Table
)
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine.base import Engine
from sqlalchemy.ext.declarative import declarative_base

from reporter.util.constant import SeqType

Base = declarative_base()


class Intake(Base):

    __tablename__ = 'intakes'

    ric = Column(String,
                 primary_key=True,
                 comment='')
    date_consumed = Column(TIMESTAMP(timezone=True), primary_key=True)
    utc_offset = Column(Integer, nullable=False)
    val = Column(Numeric(15, 6), nullable=True)

    def __init__(self,
                 ric: str,
                 date_consumed: datetime,
                 utc_offset: int,
                 val: Union[None, str]):
        
        self.ric = ric
        self.date_consumed = date_consumed
        self.utc_offset = utc_offset
        self.val = val

    def to_dict(self) -> Dict[str, Any]:
        return {'ric': self.ric,
                'date_consumed': self.date_consumed,
                'utc_offset': self.utc_offset,
                'val': self.val}


class IntakeSeq(Base):

    __tablename__ = 'intake_seqs'

    ric = Column(String,
                 primary_key=True,
                 comment='')
    seqtype = Column(String, primary_key=True)
    date_consumed = Column(TIMESTAMP(timezone=True), primary_key=True)
    vals = Column(postgresql.ARRAY(Float), nullable=True)

    def __init__(self,
                 ric: str,
                 seqtype: SeqType,
                 date_consumed: datetime,
                 vals: Union[None, List[float]]):

        self.ric = ric
        self.date_consumed = date_consumed
        self.seqtype = seqtype.value
        self.vals = vals

    def to_dict(self):
        return {'ric': self.ric,
                'date_consumed': self.date_consumed,
                'seqtype': self.seqtype,
                'vals': self.vals}


class Close(Base):

    __tablename__ = 'closes'

    ric = Column(String,
                 primary_key=True,
                 comment='Reuters Instrument Code')
    t = Column(TIMESTAMP(timezone=True), primary_key=True)

    def __init__(self, ric: str, t: datetime):

        self.ric = ric
        self.t = t

    def to_dict(self) -> Dict[str, Any]:
        return {'ric': self.ric, 't': self.t}


class Summary(Base):

    __tablename__ = 'summaries'

    summary_id = Column(String, primary_key=True)
    date_start = Column(TIMESTAMP(timezone=True), nullable=False)
    date_start = Column(TIMESTAMP(timezone=True), nullable=False)
    summary = Column(String, nullable=False)
    #isins = Column(postgresql.ARRAY(String),
                   #nullable=True,
                   #comment='International Securities Identification Number')
    #countries = Column(postgresql.ARRAY(String), nullable=True)
    #categories = Column(postgresql.ARRAY(String), nullable=True)
    #keywords_headline = Column(String, nullable=True)
    #keywords_article = Column(String, nullable=True)
    #simple_headline = Column(String, nullable=True)
    #tokens = Column(postgresql.ARRAY(String), nullable=True)
    #tag_tokens = Column(postgresql.ARRAY(String), nullable=True)
    #dictionary = Column(postgresql.UUID(as_uuid=True), nullable=True)
    ##is_used = Column(Boolean, nullable=True)
    phase = Column(String, nullable=True)

    def __init__(self,
                 summary_id: str,
                 date_start: datetime,
                 date_end: datetime,
                 summary: str,
                 phase: str):

        self.summary_id = summary_id
        self.date_start = date_start
        self.date_end = date_end
        self.summary = summary
        self.phase = phase


class Instrument(Base):

    __tablename__ = 'instruments'

    ric = Column(String,
                 primary_key=True,
                 comment='Reuters Instrument Code')
    description = Column(String)
    currency = Column(String)
    type_ = Column('type', String)
    exchange = Column(String)

    def __init__(self,
                 ric: str,
                 description: str,
                 currency: str,
                 type_: str,
                 exchange: str):

        self.ric = ric
        self.description = description
        self.currency = currency
        self.type_ = type_
        self.exchange = exchange


class HumanEvaluation(Base):

    __tablename__ = 'human_evaluation'

    article_id = Column(String, primary_key=True)
    ordering = Column(postgresql.ARRAY(String))
    fluency = Column(String)
    informativeness = Column(String)
    note = Column(String)
    is_target = Column(Boolean)

    def __init__(self,
                 article_id: str,
                 ordering: List[str],
                 fluency: Union[None, str] = None,
                 informativeness: Union[None, str] = None,
                 note: Union[None, str] = None,
                 is_target: Union[bool, None] = None):

        self.article_id = article_id
        self.ordering = ordering
        self.fluency = fluency
        self.informativeness = informativeness
        self.note = note
        self.is_target = is_target


class GenerationResult(Base):

    __tablename__ = 'generation_results'

    article_id = Column(String, primary_key=True)
    method_name = Column(String, primary_key=True)
    result = Column(String)
    correctness = Column(Integer)

    def __init__(self,
                 article_id: str,
                 method_name: str,
                 result: Union[str, None] = None):

        self.article_id = article_id
        self.method_name = method_name
        self.result = result


def create_tables(engine: Engine) -> None:
    Base.metadata.create_all(engine, tables=[Intake.__table__,
                                             IntakeSeq.__table__,
                                             Summary.__table__,
                                             Instrument.__table__,
                                             HumanEvaluation.__table__,
                                             GenerationResult.__table__])


def create_table(engine: Engine, table: Table) -> None:
    Base.metadata.create_all(engine, tables=[table])
