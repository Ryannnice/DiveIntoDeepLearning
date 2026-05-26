import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# 参数
dt = 0.1
N = 15  # 预测时域
Q = np.diag([1, 1, 0.1, 0.1])
R = np.diag([0.01, 0.01])
A = np.array([[1, 0, dt, 0],
              [0, 1, 0, dt],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])
B = np.array([[dt**2/2, 0],
              [0, dt**2/2],
              [dt, 0],
              [0, dt]])

# 初始状态与目标
x0 = np.array([0, 0, 0, 0])
x_goal = np.array([5, 5, 0, 0])

# 变量定义
x = cp.Variable((4, N+1))
u = cp.Variable((2, N))

cost = 0
constraints = [x[:, 0] == x0]

for k in range(N):
    cost += cp.quad_form(x[:, k] - x_goal, Q) + cp.quad_form(u[:, k], R)
    constraints += [x[:, k+1] == A @ x[:, k] + B @ u[:, k],
                    cp.norm(u[:, k], 'inf') <= 1.0]  # 控制输入限制

# 优化问题
prob = cp.Problem(cp.Minimize(cost), constraints)
prob.solve(solver=cp.OSQP)

# 轨迹提取
x_traj = x.value.T
plt.plot(x_traj[:, 0], x_traj[:, 1], '-o')
plt.plot(x_goal[0], x_goal[1], 'rx')
plt.xlabel("x"); plt.ylabel("y"); plt.title("MPC Trajectory")
plt.axis("equal")
plt.show()
