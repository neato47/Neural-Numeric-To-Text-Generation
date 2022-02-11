from collections import OrderedDict
from decimal import Decimal
from math import ceil
from typing import List
from xml.etree.ElementTree import fromstring

import numpy
from torch import Tensor
try:
    from torchtext.data import Batch
except ImportError:
    from torchtext.legacy.data import Batch
    

FORMULAE = OrderedDict({
    'Δ': lambda diff, _: int(diff),
    'Δ-round-down-10': lambda diff, _: int(int(diff / 10) * 10),
    'Δ-round-down-100': lambda diff, _: int(int(diff / 100) * 100),
    'Δ-round-up-10': lambda diff, _: int(ceil(diff / 10) * 10),
    'Δ-round-up-100': lambda diff, _: int(ceil(diff / 100) * 100),
    'z': lambda _, latest: int(latest),
    'z-round-down-100': lambda _, latest: int(int(latest / 100) * 100),
    'z-round-down-1000': lambda _, latest: int(int(latest / 1000) * 1000),
    'z-round-down-10000': lambda _, latest: int(int(latest / 10000) * 10000),
    'z-round-up-100': lambda _, latest: int(ceil(latest / 100) * 100),
    'z-round-up-1000': lambda _, latest: int(ceil(latest / 1000) * 1000),
    'z-round-up-10000': lambda _, latest: int(ceil(latest / 10000) * 10000)
})


def find_operation(ref_token: str,
                   prev_trading_day_close: Decimal,
                   latest: Decimal) -> str:

    results = []
    ref = Decimal(ref_token)
    diff = abs(latest - prev_trading_day_close)
    for _, formula in FORMULAE.items():
        result = abs(ref - formula(diff, latest))
        results.append(result)

    optimal_operation_id = numpy.argmin(results)
    optimal_operation_name, _ = list(FORMULAE.items())[optimal_operation_id]
    return '<yen val="{}"/>'.format(optimal_operation_name)


def perform_operation(tag: int,
                      prev_trading_day_close: float,
                      latest: float) -> str:
    operation_name = fromstring(tag).attrib['val']
    formula = FORMULAE.get(operation_name, 'z')
    diff = abs(latest - prev_trading_day_close)
    return str(formula(diff, latest)) + '円'


def replace_tags_with_vals(tokens: List[str],
                           prev_trading_day_close: float,
                           latest: float) -> List[str]:
    results = []
    for token in tokens:
        if token.startswith('<yen val="') and token.endswith('"/>'):
            results.append(perform_operation(token, prev_trading_day_close, latest))
        else:
            results.append(token)
    return results


def get_latest_closing_vals(batch: Batch,
                            raw_long_field: str,
                            times: Tensor) -> List[int]:
    # TSE close hour
    closing_hour = 15
    return [x[1] if t >= closing_hour else x[0]
            for x, t in zip(getattr(batch, raw_long_field).data, times)]
