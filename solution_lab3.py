# -*- coding: utf-8 -*-
"""
Лабораторная работа №3. Метод Рунге-Кутта. Вариант 1.
Задача 1: y'' + 4y' + 4y = 0, y(0)=1, y'(0)=-1 на [0;1], шаги 0.1, 0.01, 0.001.
Задача 2: y' = -2y + x^2, y(0)=1, метод Рунге-Кутта с адаптивным шагом.
"""
import numpy as np
import matplotlib.pyplot as plt

# ================= Задача 1 =================
# Сведение к системе первого порядка: u1=y, u2=y'
#   u1' = u2
#   u2' = -4*u2 - 4*u1
def f_sys(t, u):
    return np.array([u[1], -4.0 * u[1] - 4.0 * u[0]])

def rk4_system(f, t0, t1, u0, h):
    n = int(round((t1 - t0) / h))
    ts = np.linspace(t0, t1, n + 1)
    us = np.zeros((n + 1, len(u0)))
    us[0] = u0
    u = u0.astype(float)
    for i in range(n):
        t = ts[i]
        k1 = f(t, u)
        k2 = f(t + h / 2, u + h / 2 * k1)
        k3 = f(t + h / 2, u + h / 2 * k2)
        k4 = f(t + h, u + h * k3)
        u = u + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        us[i + 1] = u
    return ts, us

# Точное решение: y = (1 + t) e^{-2t}
def y_exact(t):
    return (1.0 + t) * np.exp(-2.0 * t)

u0 = np.array([1.0, -1.0])
steps = [0.1, 0.01, 0.001]
results = {}
print("Задача 1: значение y(1) и погрешность")
print("%8s | %14s | %12s" % ("h", "y(1) числ.", "погрешность"))
for h in steps:
    ts, us = rk4_system(f_sys, 0.0, 1.0, u0, h)
    results[h] = (ts, us)
    err = abs(us[-1, 0] - y_exact(1.0))
    print("%8.3f | %14.9f | %12.2e" % (h, us[-1, 0], err))
print("Точное значение y(1) = %.9f" % y_exact(1.0))

# График решения (для разных шагов) + точное
plt.figure(figsize=(7, 5))
for h in steps:
    ts, us = results[h]
    plt.plot(ts, us[:, 0], label='h = %g' % h)
tt = np.linspace(0, 1, 500)
plt.plot(tt, y_exact(tt), 'k--', lw=1, label='точное решение')
plt.xlabel('t'); plt.ylabel('y')
plt.title('Задача 1: решение y(t) для разных шагов')
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig('fig_l3_t1_sol.png', dpi=150); plt.close()

# График сравнения скорости сходимости: погрешность по узлам (лог. шкала)
plt.figure(figsize=(7, 5))
for h in steps:
    ts, us = results[h]
    err = np.abs(us[:, 0] - y_exact(ts))
    err[err == 0] = 1e-18
    plt.semilogy(ts, err, label='h = %g' % h)
plt.xlabel('t'); plt.ylabel('|y_числ - y_точн| (лог. шкала)')
plt.title('Задача 1: сравнение погрешности для разных шагов')
plt.legend(); plt.grid(True, which='both'); plt.tight_layout()
plt.savefig('fig_l3_t1_err.png', dpi=150); plt.close()

# ================= Задача 2 =================
# y' = -2y + x^2,  y(0)=1, адаптивный шаг (метод Рунге-Кутта 4 порядка
# с контролем погрешности по правилу Рунге - удвоение/деление шага).
def f2(x, y):
    return -2.0 * y + x * x

def rk4_step(f, x, y, h):
    k1 = f(x, y)
    k2 = f(x + h / 2, y + h / 2 * k1)
    k3 = f(x + h / 2, y + h / 2 * k2)
    k4 = f(x + h, y + h * k3)
    return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

def adaptive_rk4(f, x0, x1, y0, tol=1e-6, h0=0.05):
    x, y, h = x0, y0, h0
    xs, ys, hs = [x0], [y0], []
    while x < x1 - 1e-12:
        if x + h > x1:
            h = x1 - x
        y_big = rk4_step(f, x, y, h)              # один шаг h
        y_half = rk4_step(f, x, y, h / 2)
        y_two = rk4_step(f, x + h / 2, y_half, h / 2)  # два шага h/2
        err = abs(y_two - y_big) / 15.0           # оценка по правилу Рунге
        if err <= tol or h <= 1e-5:
            x += h
            y = y_two + (y_two - y_big) / 15.0     # уточнение
            xs.append(x); ys.append(y); hs.append(h)
            if err < tol / 10 and h < 0.4:
                h *= 1.5                            # увеличиваем шаг
        else:
            h *= 0.5                                # уменьшаем шаг
    return np.array(xs), np.array(ys), np.array(hs)

def y2_exact(x):
    return 0.75 * np.exp(-2.0 * x) + 0.5 * x * x - 0.5 * x + 0.25

xs, ys, hs = adaptive_rk4(f2, 0.0, 2.0, 1.0, tol=1e-6, h0=0.05)
print("\nЗадача 2: адаптивный шаг")
print("Число шагов:", len(hs))
print("y(2) числ. = %.8f, точное = %.8f, погрешность = %.2e"
      % (ys[-1], y2_exact(2.0), abs(ys[-1] - y2_exact(2.0))))
print("Диапазон шага: h_min=%.4g, h_max=%.4g" % (hs.min(), hs.max()))

plt.figure(figsize=(7, 5))
plt.plot(xs, ys, 'o-', ms=3, label='адаптивный РК4')
xx = np.linspace(0, 2, 500)
plt.plot(xx, y2_exact(xx), 'k--', lw=1, label='точное решение')
plt.xlabel('x'); plt.ylabel('y')
plt.title('Задача 2: решение y(x) методом с адаптивным шагом')
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig('fig_l3_t2_sol.png', dpi=150); plt.close()

plt.figure(figsize=(7, 5))
plt.plot(xs[1:], hs, 'o-', ms=3, color='tab:red')
plt.xlabel('x'); plt.ylabel('величина шага h')
plt.title('Задача 2: зависимость шага h от x')
plt.grid(True); plt.tight_layout()
plt.savefig('fig_l3_t2_step.png', dpi=150); plt.close()
print("Графики сохранены.")
