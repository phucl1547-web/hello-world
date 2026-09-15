import numpy as np
import sys

print("Python:", sys.executable)
t = [1, 2, 3, 4]
y = [2, 3, 5, 6]
q = [1,1,1,1/4]
A = np.column_stack((t,np.ones_like(t)))
def cau_a():
    x, _ , rank, _ = np.linalg.lstsq(A, y , rcond=None)
    y_hat = A @ x
    r = y_hat - y
    ATR = A.T @ r
    SSE = r.T @ r
    print("\n===== CÂU A =====")
    print("x =", x)
    print("rank(A) =", rank)
    print("y_hat =", y_hat)
    print("r =", r)
    print("SSE =", SSE)
    print("A^T r =", ATR)

    print("Check x:", np.allclose(x, [1.4, 0.5]))
    print("Check SSE:", np.isclose(SSE, 0.2))
    return x
x_cu = cau_a()
def cau_b(x_cu):
    y_copy = y.copy()
    y_copy[3] = 12
    x_copy, _ , rank_copy , _ = np.linalg.lstsq(A,y_copy,rcond = None)
    y_hat_copy = A @ x_copy
    r_copy = y_hat_copy - y_copy
    ATR_copy = A.T @ r_copy
    SSE_copy = r_copy @ r_copy
    print("\n===== CÂU B =====")
    print("y_tilde =", y_copy)
    print("x_tilde =", x_copy)
    print("SSE =", SSE_copy)

    print("Hệ số góc cũ =", x_cu[0])
    print("Hệ số góc mới =", x_copy[0])

    return y_copy, x_copy
y_b,x_b = cau_b(x_cu)
def cau_c(y_b):
    sqrt_q = np.sqrt(q)
    A_c = sqrt_q[:,None] * A
    y_c = sqrt_q * y_b
    x_c,_,rank_c,_ = np.linalg.lstsq(A_c,y_c,rcond = None)
    g = A @ x_c - y_b
    h = A.T @ np.diag(q) @ g
    print("\n===== CÂU C =====")
    print(h)
    print("x_c = ", x_c)
