from typing import Any, Self, overload


class lddl:
    ''' ... '''
    def __init__(self) -> None:
        ''' ... '''
        self.dat_ld: list[dict[str, Any]] = None
        self.dat_dl: dict[str, list[Any]] = None

    @overload
    def __getitem__(self, idx: int) -> dict[str, Any]: ...

    @overload
    def __getitem__(self, idx: str) -> list[Any]: ...

    def __getitem__(self, idx: str | int) -> list[Any] | dict[str, Any]:
        ''' ... '''
        if isinstance(idx, str):
            return self.dat_dl[idx]
        elif isinstance(idx, int):
            return self.dat_ld[idx]
        else:
            raise TypeError('Unexpected key type: {}'.format(type(idx)))

    @classmethod
    def from_ld(cls, dat: list[dict[str, Any]]) -> Self:
        ''' Construct from list of dicts. '''
        obj = cls()
        obj.dat_ld = dat
        obj.dat_dl = {k: [dat[i][k] for i in range(len(dat))] for k in dat[0]}
        return obj

    @classmethod
    def from_dl(cls, dat: dict[str, list[Any]]) -> Self:
        ''' Construct from dict of lists. '''
        obj = cls()
        obj.dat_ld = [dict(zip(dat, v)) for v in zip(*dat.values())]
        obj.dat_dl = dat
        return obj
    

if __name__ == '__main__':
    ''' for testing purpose only '''
    dl = {
        'a': [0, 1, 2],
        'b': [3, 4, 5],
    }

    ld = [
        {'a': 0, 'b': 1, 'c': 2},
        {'a': 3, 'b': 4, 'c': 5},
    ]

    # test constructing from ld
    dat_ld = lddl.from_ld(ld)
    print(dat_ld.dat_ld)
    print(dat_ld.dat_dl)

    # test constructing from dl
    dat_dl = lddl.from_dl(dl)
    print(dat_dl.dat_ld)
    print(dat_dl.dat_dl)

    # test __getitem__
    print(dat_dl['a'])
    print(dat_dl[0])

    # mouse hover to check if type hints are correct
    v = dat_dl['a']
    v = dat_dl[0]