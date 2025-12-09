import numpy as np 
import cvxpy as cp
import matplotlib.pyplot as plt


def solve_cvxpy(A, B, b):
    n = A.shape[1]
    x = cp.Variable(n)

    obj = cp.Minimize(cp.quad_form(x, B))
    constraint = [A @ x == b]

    prob = cp.Problem(obj, constraint)
    prob.solve(verbose = False)

    x_star = x.value 
    obj_star = prob.value


    return obj_star, x_star

def create_B(n, seed):
    if seed is not None:
        np.random.seed(seed)

    B = np.eye(n)
    return B

def create_A(n, seed):
    if seed is not None:
        np.random.seed(seed)

    Q, R = np.linalg.qr(np.random.randn(n,n))
    vals = 1.0 + np.random.rand(n)
    A = Q @ np.diag(vals) @ Q.T


    return A


def setup(n, m, seed):
    if seed is not None:
        np.random.seed(seed)

    A = create_A(n, seed) 
    B = create_B(n, seed)

    S = A.T @ A + B

    n1 = n//3
    n2 = n//3
    n3 = n - n1 - n2

    part_1 = slice(0, n1)
    part_2 = slice(n1, n1+n2)
    part_3 = slice(n1 + n2, n1 + n2 + n3)

    S11 = S[part_1, part_1]
    S12 = S[part_1, part_2]
    S13 = S[part_1, part_3]

    S21 = S[part_2, part_1]
    S22 = S[part_2, part_2]
    S23 = S[part_2, part_3]

    S31 = S[part_3, part_1]
    S32 = S[part_3, part_2]
    S33 = S[part_3, part_3]

    fill_12 = np.zeros_like(S12)
    fill_13 = np.zeros_like(S13)
    fill_23 = np.zeros_like(S23)

    L = np.block([[S11, fill_12, fill_13], [S21, S22, fill_23], [S31, S32, S33]])

    U = S - L
    b = np.random.randn(m)

    return A, B, S, L, U, b

def objective(x, B):
    return float(x.T @ B @ x)

def constraint_violation(x, A):
    r = A @ x 
    return float(np.linalg.norm(r))

def solve (A,b):
    return np.linalg.solve(A, b)

def iterate(x, lam, A, sign_k, L, U, b):
    S = L + U

    n = x.shape[0]
    n1 = n//3
    n2 = n//3
    n3 = n - n1 - n2

    part_1 = slice(0, n1)
    part_2 = slice(n1, n1+n2)
    part_3 = slice(n1 + n2, n1 + n2 + n3)
    
    x1 = x[part_1]
    x2 = x[part_2]
    x3 = x[part_3]

    A1 = A[:, part_1]
    A2 = A[:, part_2]
    A3 = A[:, part_3]

    S11 = S[part_1, part_1]
    S12 = S[part_1, part_2]
    S13 = S[part_1, part_3]

    S21 = S[part_2, part_1]
    S22 = S[part_2, part_2]
    S23 = S[part_2, part_3]

    S31 = S[part_3, part_1]
    S32 = S[part_3, part_2]
    S33 = S[part_3, part_3]

    rhs_1 = -S12 @ x2 - S13 @ x3 - A1.T @ lam + A1.T @ b
    x1_next = np.linalg.solve(S11, rhs_1)

    rhs_2 = -S21 @ x1_next - S23 @ x3 - A2.T @ lam + A2.T @ b
    x2_next = np.linalg.solve(S22, rhs_2)

    rhs_3 = -S31 @ x1_next - S32 @ x2_next - A3.T @ lam + A3.T @ b
    x3_next = np.linalg.solve(S33, rhs_3)

    x_next = np.concatenate([x1_next, x2_next, x3_next])
    lam_next = lam + sign_k * (A @ x_next - b)

    return x_next, lam_next

def run_iterations(A, B, L, U, b, max_iter = 300, eps = 1e-4):
    optimal_obj, optimal_x = solve_cvxpy(A, B, b)
    n = A.shape[1]
    m = A.shape[0]

    x = np.random.randn(n)
    lam = np.random.randn(m)

    storage = {"x" : [], "lam" : [], "obj" : [], "sign" : [], "residual_obj" : [], "residual_x" : []}

    for i in range (max_iter):

        if i % 2 == 0:
            sign_k = 1.0
        else:
            sign_k = -1.0

        storage["sign"].append(sign_k)

        x, lam = iterate(x, lam, A, sign_k, L, U, b)
        storage["x"].append(x.copy())
        storage["lam"].append(lam.copy())
        storage["obj"].append(objective(x, B))
        storage["residual_obj"].append(abs(objective(x, B) - optimal_obj))
        storage["residual_x"].append(np.linalg.norm(x - optimal_x))

        print(f"Iter {i:03d} | sign={sign_k:+.0f} | obj={objective(x, B):.6f}")
        print(f"x = {x}")
        print(f"λ = {lam}\n")

        if abs(objective(x, B) - optimal_obj) < eps:
            break

    return storage

def M(A, L, U):
    n = A.shape[1]
    m = A.shape[0]

    S = L + U
    L_inv = np.linalg.inv(L)
    top_left = np.eye(n) - L_inv @ S
    bottom_left = A @ L_inv @ S - A
    top_right = L_inv @ A.T
    bottom_right = np.eye(m) - A @ L_inv @ A.T

    M = np.block([[top_left, top_right], [bottom_left, bottom_right]])

    return M

def M_(A, L, U):
    n = A.shape[1]
    m = A.shape[0]
    
    S = L + U
    L_inv = np.linalg.inv(L)
    top_left = np.eye(n) - L_inv @ S
    bottom_left = A - A @ L_inv @ S
    top_right = L_inv @ A.T
    bottom_right = np.eye(m) + A @ L_inv @ A.T
    
    M_ = np.block([[top_left, top_right], [bottom_left, bottom_right]])

    return M_

def MM_ (A, L, U):
    return M(A, L, U) @ M_(A, L, U)

def spectral_radius(A):
    eigenvalues = np.linalg.eigvals(A)
    abs_eig = np.abs(eigenvalues)
    return np.max(abs_eig), eigenvalues

def run_experiment(n, m, seed, max_iter, eps):
    A, B, S, L, U, b = setup(n, m, seed)
    run = run_iterations(A, B, L, U, b, max_iter, eps)

    obj_res = run["residual_obj"]
    x_res = run["residual_x"]
    

    if obj_res[-1] < eps:
        convergence = True
    else:
        convergence = False

    mm_ = MM_(A, L, U)

    rho, eigs = spectral_radius(mm_)
    return rho, eigs, len(run["x"]), convergence, obj_res, x_res

def pad_array(array):
    padded_list = list(array)
    final_value = array[-1]
    append_length = max_length - len(array)
    for i in range(append_length):
        padded_list.append(final_value)
    
    return padded_list

    
num_trials = 30
n = 15
m = 15
max_iter = 10000
eps = 1e-5
rhos = []
eigs = []
iterations = []
convergence_TF = []
full_obj_res = []
full_x_res = []

for i in range(1, num_trials + 1):
    seed = np.random.randint(0, 2**30)
    rho, eigvals, iters, convergence, obj_res, x_res = run_experiment(n, m, seed, max_iter, eps)

    rhos.append(rho)
    eigs.append(eigvals)
    iterations.append(iters)
    convergence_TF.append(convergence)
    full_obj_res.append(obj_res)
    full_x_res.append(x_res)


print(rhos)
print(convergence_TF)


max_length = 0
for j in range(len(full_obj_res)):
    max_length = max(max_length, len(full_obj_res[j]))

obj_vals = []
x_norm_vals = []

for k in range(len(full_obj_res)):
    padded_obj_res = pad_array(full_obj_res[k])
    obj_vals.append(padded_obj_res)

for l in range(len(full_x_res)):
    padded_x_res = pad_array(full_x_res[l])
    x_norm_vals.append(padded_x_res)


obj_mat = np.array(obj_vals)
x_mat = np.array(x_norm_vals)

mean_obj = np.zeros(max_length)
for n in range(max_length):
    added = 0.0
    for m in range(len(obj_mat)):
        added += obj_mat[m][n]
    mean_obj[n] = added/len(obj_mat)

mean_x_norm = np.zeros(max_length)
for n in range(max_length):
    added = 0.0
    for m in range(len(x_norm_vals)):
        added += x_mat[m][n]
    mean_x_norm[n] = added/len(x_norm_vals)



plt.figure(figsize = (10, 10))
plt.semilogy(mean_obj, linewidth = 2, label = "Average objective value residual")
plt.xlabel("Iteration Number", fontsize = 16)
plt.ylabel("Residual", fontsize = 16)
#plt.title("Average Objective Residual", fontsize = 18)
plt.savefig("Average Objective Residual.pdf", bbox_inches = "tight")

plt.figure(figsize = (10, 10))
plt.semilogy(mean_x_norm, linewidth = 2, label = "Average x norm differenc from x*")
plt.xlabel("Iteration Number", fontsize = 16)
plt.ylabel("Residual", fontsize = 16)
#plt.title("Average X Norm Residual",fontsize = 18)
plt.savefig("Average XNorm Residual.pdf", bbox_inches = "tight")







    # M = np.random.randn(n,n)
    # B = M @ M.T

    # evals, evecs = np.linalg.eigh(B)
    # evals = np.clip(evals,1e-3, None)
    # return evecs @ np.diag(evals) @ evecs.T
