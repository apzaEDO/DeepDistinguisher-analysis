"""
Compute the probability that a random generator matrix contains a pairwise XOR
violation.

More precisely, for the parameters of a binary Goppa code (n,m,t), we consider
a random matrix \(A \in \mathbb{F}_2^{k times (n-k)}\), where k = n - mt.
The function estimates the probability that there exists a pair of distinct rows
A_i, A_j such that

    w_H(A_i + A_j) < 2t - 1.

This condition corresponds to a violation of the minimum-distance constraint
expected for the non-systematic block A of a systematic generator matrix
G = (I_k | A).
"""

from math import comb


def calculate_pxor(n, m, t):
    """
    Compute the probability pxor.

    Parameters:
    n (int): Code length.
    m (int): Extension degree.
    t (int): degree of the Goppa polynomial.

    Returns:
    float: The probability that the XOR condition is satisfied.
    """
    N = m * t
    total_sum = 0

    # Sum binomial coefficients C(N, i) for i = 0, ..., 2t - 2
    for i in range(0, 2 * t - 1):
        total_sum += comb(N, i)

    # Normalize by the total number of binary vectors of length N
    return total_sum / (2 ** N)


def second_moment(n, m, t):
    """
    Compute the second-moment approximation.

    Parameters:
    n (int): Code length.
    m (int): Extension degree.
    t (int): degree of the Goppa polynomial.

    Returns:
    float: Second-moment probability estimate.
    """
    k = n - m * t

    # Number of unordered pairs among k elements
    number_of_pairs = comb(k, 2)

    pxor = calculate_pxor(n, m, t)

    return (number_of_pairs * pxor) / (
        1 + (number_of_pairs - 1) * pxor
    )


def matrix_probability(n, m, t):
    """
    Compute the probability that at least one pair satisfies the XOR condition.

    Parameters:
    n (int): Code length.
    m (int): Extension degree.
    t (int): degree of the Goppa polynomial.

    Returns:
    float: Probability that at least one valid pair exists.
    """
    k = n - m * t

    # Probability for one pair
    pxor = calculate_pxor(n, m, t)

    # Number of unordered pairs among k elements
    number_of_pairs = comb(k, 2)

    # Probability that at least one pair satisfies the condition
    return 1 - (1 - pxor) ** number_of_pairs


m = 6

for t in range(2, 9):
    print(f"T{t}=")

    for n in range(t * 8, 65, 8):
        probability = matrix_probability(n, m, t)

        print(f"Probability :{probability:.2f},")