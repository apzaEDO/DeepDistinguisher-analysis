# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from sage.coding.goppa_code import GoppaCode, _columnize
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from logging import getLogger


logger = getLogger()


class qGoppaCode(GoppaCode):
    """
    Subclass of sage's GoppaCode to allow for prime power base fields
    """

    def __init__(self, generating_pol, defining_set, base_field=None):
        if base_field is None:
            base_field = generating_pol.base_ring().prime_subfield()
            logger.info(
                f"Base field is not specified, using prime subfield {base_field}"
            )
        ext_field = generating_pol.base_ring()
        assert base_field.is_subring(
            ext_field
        ), f"{base_field} is not subfield of {ext_field}"
        self._field = base_field
        self._length = len(defining_set)
        self._generating_pol = generating_pol
        self._defining_set = defining_set
        super(GoppaCode, self).__init__(
            self._field, self._length, "GoppaEncoder", "Syndrome"
        )

        if not generating_pol.is_monic():
            raise ValueError("generating polynomial must be monic")
        F = self._field
        if not F.is_field() or not F.is_finite():
            raise ValueError(
                "generating polynomial must be defined over a finite field"
            )
        for a in defining_set:
            if generating_pol(a) == 0:
                raise ValueError(
                    "defining elements cannot be roots of generating polynomial"
                )

    def parity_check_matrix(self):

        g = self._generating_pol
        F = g.base_ring()
        n = self._length
        d = g.degree()
        alpha = self._defining_set[0]

        D = self._defining_set
        h = [(g(D[i]).inverse_of_unit()) for i in range(n)]

        # assemble top row
        M = _columnize(alpha)
        for i in range(n):
            v = _columnize(h[i])
            M = M.augment(v)
        M = M.delete_columns([0])
        old = M

        for t in range(1, d):
            # assemble row
            M = _columnize(alpha)
            for i in range(n):
                v = _columnize(h[i] * (D[i] ** t))
                M = M.augment(v)
            M = M.delete_columns([0])
            new = M
            old = old.stack(new)

        return old
