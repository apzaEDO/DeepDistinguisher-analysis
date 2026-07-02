# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
from scipy.linalg import circulant
from logging import getLogger
import torch
import argparse

from time import time
from sage.all import *
from sage.all import GF, set_random_seed
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.coding.linear_code import LinearCode
from sage.matrix.constructor import random_matrix
from sage.coding.grs_code import GeneralizedReedSolomonCode
from sage.coding.subfield_subcode import SubfieldSubcode
from sage.rings.finite_rings.integer_mod import IntegerMod_int
from src.utils import GoppaCodeParams, bool_flag
from src.data.datasource import ICodeDataSource
from src.data.codes import qGoppaCode as GoppaCode


logger = getLogger()


def parse_bike_args(unknown_args, namespace):
    parser = argparse.ArgumentParser(description="MDPC Code arguments for BIKE")

    parser.add_argument("--r", type=int, help="degree of x^r - 1")
    parser.add_argument("--w", type=int, help="weight of h0 and h1")
    parser.add_argument("--qc", type=bool_flag, help="Quasi-Cyclic or just MDPC")

    return parser.parse_args(unknown_args, namespace=namespace)


def parse_alternant_args(unknown_args, namespace):
    parser = argparse.ArgumentParser(description="Goppa&Alternant Code arguments")

    parser.add_argument("--m_alt", type=int, help="defines modulus q = 2^m")
    parser.add_argument("--t_alt", type=int, help="degree of irreducible polynomial g")
    parser.add_argument("--beta_dist", type=str, default="uniform")
    parser.add_argument("--alpha_dist", type=str, default="uniform")

    return parser.parse_args(unknown_args, namespace=namespace)


parse_fn = {
    "alternant": parse_alternant_args,
    "goppa": parse_alternant_args,
    "qc": parse_bike_args,
    "mdpc": parse_bike_args,
    "random": parse_alternant_args,
}


def to_integer(x):
    if isinstance(x, IntegerMod_int):
        return x.lift()
    elif isinstance(x, (int, np.integer, Integer)):
        return x
    return x.to_integer()


to_integer_np = np.vectorize(to_integer)


DTYPE_MAP = {
    256: np.uint8,
    65536: np.uint16,
    4294967296: np.uint32,
}


class ICodeGenDataSource(ICodeDataSource):
    def __init__(self, params, loggr=None, worker=None, verbose=True, **kwargs):
        self.worker = worker
        self.verbose = verbose
        self.set_dtype(params.Q)
        self.set_constants(params)

        self.logger = loggr
        if loggr is None:
            self.logger = logger

        self.init_rng = False

    def set_constants(self, params):
        raise NotImplementedError("This method should be implemented in subclasses")

    def set_dtype(self, q):
        """Select smallest uint type for storing codeword whose coefficients are in Fq"""
        self.dtype = np.uint64
        for max_val, dtype in DTYPE_MAP.items():
            if q - 1 < max_val:
                self.dtype = dtype
                break

    def _initialize_rng(self):
        """Set a seed for randomness"""
        worker_seed = self.worker
        if worker_seed is None:
            # if generating during training, dataloader sets seed for numpy rng,
            # we use numpy as seed
            worker = torch.utils.data.get_worker_info()
            if worker:
                self.worker = worker_seed = worker.id
            else:
                worker_seed = np.random.randint(1 << 40)

        G_seed = hash((self.seed, worker_seed, int(time()))) % (1 << 30)
        if self.verbose:
            logger.info(
                f"[{self.worker} ([{self.seed}])][{self.__class__.__name__}] Worker seed set to {G_seed}"
            )
        set_random_seed(G_seed)
        np.random.seed(G_seed)
        self.init_rng = True


class CodeGenerator(ICodeDataSource):
    MAX_ATTEMPTS = 1000

    def __init__(self, params, loggr=None, worker=None, verbose=True, **kwargs):
        self.worker = worker
        self.verbose = verbose
        self.set_dtype(params.Q)
        self.set_constants(params)

        self.logger = loggr
        if loggr is None:
            self.logger = logger

        self.init_rng = False

    def set_constants(self, params):
        self.q = params.Q
        self.n = params.code_len
        self.t = params.t_alt
        if hasattr(params, "code_dim") and params.code_dim > 0:
            self.k = params.code_dim
        elif hasattr(params, "k") and params.k > 0:
            self.k = params.k
        elif hasattr(params, "r") and params.r > 0:
            self.k = params.r
        else:
            self.k = params.code_len - params.m_alt * params.t_alt
        self.seed = params.seed
        self.F = GF(self.q)
        self.standard_only = (
            params.standard_only if hasattr(params, "standard_only") else False
        )
        self.I = np.eye(self.k, dtype=int)
       

    def _initialize_rng(self):
        """Set a seed for randomness"""
        worker_seed = self.worker
        if worker_seed is None:
            # if generating during training, dataloader sets seed for numpy rng,
            # we use numpy as seed
            worker = torch.utils.data.get_worker_info()
            if worker:
                self.worker = worker_seed = worker.id
            else:
                worker_seed = np.random.randint(1 << 40)

        G_seed = hash((self.seed, worker_seed, self.n, self.k, int(time()))) % (1 << 30)
        if self.verbose:
            logger.info(
                f"[{self.worker} ([{self.k,self.n}])][{self.__class__.__name__}] Worker seed set to {G_seed}"
            )
        set_random_seed(G_seed)
        np.random.seed(G_seed)
        self.init_rng = True

    def read_item(self, index):
        return self.generate_code(serialize=False)

    def parity_check_from_systematic_G(G, k):
        # G: (k x n) binaire (0/1), systématique [I_k | A]
        n = G.shape[1]
        r = n - k
        A = G[:, k:] & 1
        H = np.concatenate([A.T, np.eye(r, dtype=np.int64)], axis=1) & 1
    
        # sanity check: G H^T = 0 mod 2
        assert ((G @ H.T) & 1).sum() == 0
        return H

        
    def generate_code_G(self, serialize=True, lift=True):
        """
        Generate a code generator matrix, we take the systematic form and omit the identity part.
        if rank G > k, try again.

        """
        if not self.init_rng:
            self._initialize_rng()

        attempt = 0
        while attempt < self.MAX_ATTEMPTS:
            attempt += 1

            C, parameters = self.get_code()

            if C.dimension() != self.k:
                logger.info(f"Expected dimension {self.k} got {C.dimension()}")
                continue
            if self.standard_only:
                G = C.systematic_generator_matrix().numpy()
                # check if in standard form
                GI = G[:, : self.k].astype(int)
                if (GI != self.I).any():
                    continue
            else:
                C_std, _ = C.standard_form(False)
                G = C_std.systematic_generator_matrix().numpy()

            return self._serialize_data(
                G,
                serialize=serialize,
                lift=lift,
                identity_width=self.k,
                **parameters,
            )

        raise ValueError(f"Tried {attempt} times, Invalid Parameters n, k, t")

    def generate_code_GD(self, serialize=True, lift=True):
        """
        Generate a systematic generator matrix G = (I_k | A).
        Optionally keep only instances whose A-part has total Hamming weight
        in [self.weight_min, self.weight_max].
        """
        if not self.init_rng:
            self._initialize_rng()
    
        weight_min=312 #getattr(self, "weight_min", None)
        weight_max=336 #getattr(self, "weight_max", None)
    
        attempt = 0
        while attempt < self.MAX_ATTEMPTS:
            attempt += 1
    
            C, parameters = self.get_code()
    
            if C.dimension() != self.k:
                logger.info(f"Expected dimension {self.k} got {C.dimension()}")
                continue
    
            if self.standard_only:
                G = C.systematic_generator_matrix().numpy()
                GI = G[:, : self.k].astype(int)
                if (GI != self.I).any():
                    continue
            else:
                C_std, _ = C.standard_form(False)
                G = C_std.systematic_generator_matrix().numpy()
    
            # ---- filtre de poids sur A ----
            A = G[:, self.k:]
            wt_A = np.count_nonzero(A)
    
            if weight_min is not None and wt_A < weight_min:
                continue
            if weight_max is not None and wt_A > weight_max:
                continue
    
            return self._serialize_data(
                G,
                serialize=serialize,
                lift=lift,
                identity_width=self.k,
                **parameters,
            )
    
        raise ValueError(f"Tried {attempt} times, Invalid Parameters n, k, t")

    def generate_code_GT(self, serialize=True, lift=True):
        """
        Generate a systematic generator matrix G = (I_k | A) and keep only
        instances whose rows in A all have Hamming weight >= 2*t.
        If serialize=True, only A is returned.
        """
        if not self.init_rng:
            self._initialize_rng()
    
        attempt = 0
        while attempt < self.MAX_ATTEMPTS*10:
            attempt += 1
    
            C, parameters = self.get_code()
    
            if C.dimension() != self.k:
                logger.info(f"Expected dimension {self.k} got {C.dimension()}")
                continue
    
            if self.standard_only:
                G = C.systematic_generator_matrix().numpy()
                GI = G[:, : self.k].astype(int)
                if (GI != self.I).any():
                    continue
            else:
                C_std, _ = C.standard_form(False)
                G = C_std.systematic_generator_matrix().numpy()
    
            A = G[:, self.k:]
            row_weights_A = np.count_nonzero(A, axis=1)
            if np.any(row_weights_A < 2 * self.t):
                continue

            
            
            return self._serialize_data(
                G,
                serialize=serialize,
                lift=lift,
                identity_width=self.k,
                **parameters,
            )
    
        raise ValueError(f"Tried {attempt} times, Invalid Parameters n, k, t")

    def generate_code_H(self, serialize=True, lift=True):
        if not self.init_rng:
            self._initialize_rng()
    
        r = self.n - self.k
        attempt = 0
    
        while attempt < self.MAX_ATTEMPTS:
            attempt += 1
    
            C, parameters = self.get_code()
    
            if C.dimension() != self.k:
                logger.info(f"Expected dimension {self.k} got {C.dimension()}")
                continue
    
            H = C.parity_check_matrix()   # garder la matrice Sage ici
            H_left = H.matrix_from_columns(range(r))
    
            if H_left.rank() != r:
                continue
    
            # mise en forme systématique par opérations de lignes seulement
            H = (H_left.inverse() * H).numpy()
            
            return self._serialize_data(
                H,
                serialize=serialize,
                lift=lift,
                identity_width=r,
                **parameters,
            )
    
        raise ValueError(f"Tried {attempt} times, Invalid Parameters n, k, t")

    def generate_code_HT(self, serialize=True, lift=True):
        if not self.init_rng:
            self._initialize_rng()
    
        attempt = 0
        while attempt < self.MAX_ATTEMPTS*10:
            attempt += 1
    
            C, parameters = self.get_code()
    
            if C.dimension() != self.k:
                logger.info(f"Expected dimension {self.k} got {C.dimension()}")
                continue
    
            if self.standard_only:
                G = C.systematic_generator_matrix().numpy()
                GI = G[:, : self.k].astype(int)
                if (GI != self.I).any():
                    continue
            else:
                C_std, _ = C.standard_form(False)
                G = C_std.systematic_generator_matrix().numpy()
    
            A = G[:, self.k:]
            row_weights_A = np.count_nonzero(A, axis=1)
            if np.any(row_weights_A < 2 * self.t):
                continue

            H = A.transpose().augment(sa.identity_matrix(GF(2), self.n-self.k))
            
            return self._serialize_data(
                G,
                serialize=serialize,
                lift=lift,
                identity_width=self.k,
                **parameters,
            )
    
        raise ValueError(f"Tried {attempt} times, Invalid Parameters n, k, t")

        
    def get_code(self):
        G = random_matrix(self.F, self.k, self.n)
        return LinearCode(G), dict()

    def get_code_t(self):
        lignes = []
        for i in range(self.k):
            poids = True
            while poids :
                ligne = random_vector(self.F,self.n)
                w = sum(ligne.list())
                if w>= self.t*2 :
                    lignes.append(ligne)
                    poids = False
        G = matrix(self.F,lignes)
        return LinearCode(G),dict()

    def _serialize_data(self, M, serialize=True, lift=True, identity_width=None, **kwargs):
        if serialize:
            if identity_width is None:
                raise ValueError("identity_width must be provided when serialize=True")
            M = M[:, identity_width:]
    
        if not lift:
            kwargs["G"] = M
            return kwargs
    
        data = self._lift(M, **kwargs)
    
        for key, value in data.items():
            data[key] = value.astype(self.dtype)
    
        return data

    def _lift(self, G, **kwargs):
        data = dict(G=to_integer_np(G))
        for key, value in kwargs.items():
            elements = np.array(list(map(list, value)))
            elements = to_integer_np(elements)
            data[key] = elements

        return data

    def set_dtype(self, q):
        """Select smallest uint type for storing generator matrix whose coefficients are in Fq"""
        self.dtype = np.uint64
        for max_val, dtype in DTYPE_MAP.items():
            if q - 1 < max_val:
                self.dtype = dtype
                break


class RandomQCGenerator(CodeGenerator):
    def __init__(self, params, size):
        self.size = size
        super().__init__(params)

    def set_constants(self, params):
        self.q = params.Q
        self.n = params.code_len
        self.r = params.r
        self.k = params.r
        self.seed = params.seed
        self.F = GF(self.q)

        self.cache = [None] * self.size

    def read_item(self, index):
        index = index % self.size
        if self.cache[index] is None:
            self.cache[index] = self.generate_code()

        return {"h": self.cache[index]}

    def generate_code(self):
        """
        Generate a qc code
        """
        if not self.init_rng:
            self._initialize_rng()

        return np.random.randint(0, self.q, size=self.r, dtype=np.int32)


class AlternantCodeGenerator(CodeGenerator):
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self.set_constants(params)

    def set_constants(self, params):
        super().set_constants(params)
        self.m = params.m_alt
        assert self.q**self.m >= self.n, f"n={self.n} must be smaller than q=2^m"
        self.t = params.t_alt
        self.k = self.n - self.m * self.t
        self.grs_dim = self.t

        self.alpha_dist = params.alpha_dist
        self.beta_dist = params.beta_dist
        if self.beta_dist not in ("uniform", "fixed"):
            self.p = float(self.beta_dist)
            assert 0 <= self.p and self.p <= 1
            self.prior_betas = []

        if is_prime(self.q):
            self.Fe = GF(self.q**self.m)
        else:
            Re = PolynomialRing(GF(self.q), "y")
            h = Re.irreducible_element(self.m, algorithm="first_lexicographic")
            logger.info(f"Using polynomial {h} for F_{self.q}^{self.m} extension")
            self.Fe = Re.quotient(h)

    def get_code(self):
        alpha = self.get_alpha(self.Fe, self.n)
        assert len(alpha) == self.n
        beta = self.get_beta(self.n)

        GRS_Code = GeneralizedReedSolomonCode(alpha, self.grs_dim, beta).dual_code()
        return SubfieldSubcode(GRS_Code, self.F), dict(alpha=alpha, beta=beta)

    def _initialize_rng(self):

        if self.beta_dist == "fixed":
            # Generate a common polynomial g first
            set_random_seed(self.seed)
            np.random.seed(self.seed)
            self.beta = self.generate_beta_uniform(self.t)
            if self.verbose:
                logger.info(f"Using beta/g = {self.beta}")

        if self.alpha_dist == "fixed":
            set_random_seed(self.seed)
            np.random.seed(self.seed)
            self.alpha = self.sample_distinct_elements(self.Fe, self.n)
            if self.verbose:
                logger.info(f"Using fixed permuation of alpha: {self.alpha[:6]}...")

        super()._initialize_rng()

    def get_beta(self, n):
        if self.beta_dist == "fixed":
            return self.beta
        elif self.beta_dist == "uniform":
            beta = self.generate_beta_uniform(n)
        else:
            beta = self.generate_beta_unbalanced(n)

        return beta

    def get_alpha(self, Fe, n):
        if self.alpha_dist == "fixed":
            return self.alpha
        else:
            return self.sample_distinct_elements(Fe, n)

    def generate_beta_uniform(self, n):
        column_multipliers = []

        while len(column_multipliers) < n:
            multiplier = self.Fe.random_element()
            if multiplier != 0:
                column_multipliers.append(multiplier)

        return column_multipliers

    def generate_beta_unbalanced(self, n):
        return self.generate_beta_uniform(n)

    def sample_distinct_elements(self, F, n):
        if F.order() < 1 << 22:
            return np.random.choice(list(F), n, replace=False)

        elements = []
        while len(elements) < n:
            r = F.random_element()
            if r not in elements:
                elements.append(r)
        return elements


class GoppaCodeGenerator(AlternantCodeGenerator):

    def get_code(self):
        alpha = self.get_alpha(self.Fe, self.n)
        g = self.get_beta(self.t)
        return GoppaCode(g, alpha, self.F), dict(alpha=alpha, beta=g.list())

    def generate_beta_uniform(self, degree):
        # Define the polynomial ring over the finite field
        R = PolynomialRing(self.Fe, "x")

        # Define an irreducible polynomial (Goppa polynomial)
        if hasattr(R, "irreducible_element"):
            return R.irreducible_element(degree, algorithm="random")

        x = R.gen()
        while True:
            f = x**degree + R.random_element(degree=(0, degree - 1))
            if f.is_irreducible():
                return f

    def generate_beta_unbalanced(self, n):

        if len(self.prior_betas) > 0 and np.random.random() < self.p:
            # reuse samples
            ig = np.random.randint(len(self.prior_betas))
            # logger.info(f"{ig} {len(self.prior_betas)}")
            return self.prior_betas[ig]
        else:
            g = self.generate_beta_uniform(n)
            self.prior_betas.append(g)
            return g


class MDPCCodeGenerator(CodeGenerator):

    def set_constants(self, params):

        params.code_len = 2 * params.r
        params.code_dim = params.r
        params.k = params.r

        super().set_constants(params)

        self.r = params.r
        self.w = params.w

    def get_code(self):
        H = self.mdpc_matrix(self.r * 2, self.r, self.w)
        return LinearCode(H), dict()

    def mdpc_matrix(self, n, k, w):
        H = Matrix(GF(2), n - k, n)
        # Fill each row with exactly w ones
        for i in range(n - k):
            # Randomly select w unique positions for ones in the row
            ones_positions = sample(range(n), w)
            for pos in ones_positions:
                H[i, pos] = 1

        return H


class QCCodeGenerator(MDPCCodeGenerator):

    def set_constants(self, params):
        super().set_constants(params)
        # Define the polynomial ring over GF(2)
        R = PolynomialRing(self.F, "x")
        x = R.gen()

        # Define the modulus polynomial
        modulus = x**self.r - 1
        self.qR = R.quotient(modulus, "xb")

    def generate_code(self, serialize=True, lift=True):

        h1 = self.sample_element_w()
        attempt = 0
        while attempt < self.MAX_ATTEMPTS:
            attempt += 1
            h0 = self.sample_element_w()

            if h0.is_unit():
                h0_inverse = h0.inverse_of_unit()
                h = h0_inverse * h1
                return self._serialize_data(serialize, lift, h0=h0, h1=h1, h=h)
        self.logger.error(f"Max Attempts {self.MAX_ATTEMPTS} reached!")
        exit()

    def sample_element_w(self):
        coeffs = np.zeros(self.r, dtype=int)
        positions = np.random.choice(self.r, self.w // 2, replace=False)
        coeffs[positions] = 1

        return self.qR(coeffs.tolist())

    def _serialize_data(self, serialize, lift, **kwargs):
        if not lift:
            return kwargs

        lifted_data = {}
        for key, value in kwargs.items():
            lifted_data[key] = np.array(value.list(), dtype=self.dtype)
        return lifted_data


class GoppaAllGenerator(ICodeDataSource):
    def __init__(self, params, size, keys, is_random=False):
        self.max_code_len = 0
        _sets = GoppaCodeParams(params.param_sets)
        assert len(_sets) > 0, "No valid parameter set"
        self.sample_t_uniform = True
        self.datasources = {}

        _cls = GoppaCodeGenerator
        if is_random:
            _cls = CodeGenerator

        for i, _set in enumerate(_sets):
            n, m, t = _set
            this_params = deepcopy(params)
            this_params.code_len = n
            this_params.code_dim = n - m * t
            this_params.m_alt = m
            this_params.t_alt = t
            self.max_code_len = max(self.max_code_len, n)
            gen = _cls(this_params, verbose=i < 2)
            self.datasources[_set] = gen

        self._sets = _sets
        self._set_data_size(size)

    def _set_data_size(self, size):
        self.size = size

    def read_item(self, index):

        if self.sample_t_uniform:
            _set = self._sets.sample(index)
        else:
            _set = self._sets[index % len(self._sets)]

        item = self.datasources[_set].read_item(index // len(self._sets))
        item["set"] = _set
        return item
