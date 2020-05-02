from ranger.radam import RAdam
from ranger.lookahead import Lookahead


def ranger(params):
    radam = RAdam(params)
    lookahead = Lookahead(radam)
    return lookahead
