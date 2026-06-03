import numpy as np
import matplotlib.pyplot as plt

# Настраиваем стиль графиков (крупные шрифты как на предыдущих)
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
})

# ============================================================
# ГРАФИК 3: Доступное время (часы в неделю)
# ============================================================
# Параметры функций принадлежности из fuzzy_engine.py:
# "мало": трапецеидальная [0, 0, 2, 4]
# "умеренно": треугольная [2, 4, 6]
# "много": трапецеидальная [4, 7, 15, 20]

x_time = np.linspace(0, 20, 1000)

# "мало" — трапецеидальная [0, 0, 2, 4]
mu_little = np.where(x_time <= 0, 1.0,
              np.where(x_time <= 2, 1.0,
              np.where(x_time <= 4, 1 - (x_time - 2) / 2, 0)))

# "умеренно" — треугольная [2, 5, 8]
mu_moderate = np.where(x_time < 2, 0,
                np.where(x_time <= 5, (x_time - 2) / 3,
                np.where(x_time <= 8, (8 - x_time) / 3, 0)))

# "много" — трапецеидальная [6, 8, 15, 20]
mu_many = np.where(x_time < 6, 0,
            np.where(x_time <= 8, (x_time - 6) / 2,
            np.where(x_time <= 15, 1.0,
            np.where(x_time <= 20, (20 - x_time) / 5, 0))))

fig1, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(x_time, mu_little, 'b-', linewidth=2.5, label='«мало»')
ax1.plot(x_time, mu_moderate, 'g--', linewidth=2.5, label='«умеренно»')
ax1.plot(x_time, mu_many, 'r-.', linewidth=2.5, label='«много»')

# Заштрихованные области под кривыми (как на предыдущих графиках)
ax1.fill_between(x_time, 0, mu_little, alpha=0.15, color='blue')
ax1.fill_between(x_time, 0, mu_moderate, alpha=0.15, color='green')
ax1.fill_between(x_time, 0, mu_many, alpha=0.15, color='red')

ax1.set_xlabel('Часы в неделю', fontsize=14)
ax1.set_ylabel('Степень принадлежности μ', fontsize=14)
ax1.set_title('Функции принадлежности лингвистической переменной «Доступное время»', fontsize=16)
ax1.set_xlim(0, 20)
ax1.set_ylim(0, 1.05)
ax1.set_xticks(np.arange(0, 21, 2))
ax1.set_yticks(np.arange(0, 1.1, 0.2))
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')

plt.tight_layout()
plt.savefig('figure3_time.png', dpi=300, bbox_inches='tight')
print('✅ График 3 сохранён: figure3_time.png')

# ============================================================
# ГРАФИК 4: Карьерная направленность (шкала 0.0–1.0)
# ============================================================
# Параметры функций принадлежности из fuzzy_engine.py:
# "личное развитие": треугольная [0.0, 0.0, 0.5]
# "смешанная": треугольная [0.0, 0.5, 1.0]
# "трудоустройство": треугольная [0.5, 1.0, 1.0]

x_career = np.linspace(0, 1, 1000)

# "личное развитие" — треугольная [0.0, 0.0, 0.5]
mu_personal = np.where(x_career <= 0, 1.0,
                np.where(x_career <= 0.5, 1 - x_career / 0.5, 0))

# "смешанная" — треугольная [0.0, 0.5, 1.0]
mu_mixed = np.where(x_career < 0, 0,
             np.where(x_career <= 0.5, x_career / 0.5,
             np.where(x_career <= 1.0, 1 - (x_career - 0.5) / 0.5, 0)))

# "трудоустройство" — треугольная [0.5, 1.0, 1.0]
mu_employment = np.where(x_career < 0.5, 0,
                  np.where(x_career <= 1.0, (x_career - 0.5) / 0.5, 1.0))

fig2, ax2 = plt.subplots(figsize=(10, 6))

ax2.plot(x_career, mu_personal, 'b-', linewidth=2.5, label='«личное развитие»')
ax2.plot(x_career, mu_mixed, 'g--', linewidth=2.5, label='«смешанная»')
ax2.plot(x_career, mu_employment, 'r-.', linewidth=2.5, label='«трудоустройство»')

# Заштрихованные области под кривыми
ax2.fill_between(x_career, 0, mu_personal, alpha=0.15, color='blue')
ax2.fill_between(x_career, 0, mu_mixed, alpha=0.15, color='green')
ax2.fill_between(x_career, 0, mu_employment, alpha=0.15, color='red')

ax2.set_xlabel('Карьерная направленность (0 — личное развитие, 1 — трудоустройство)', fontsize=13)
ax2.set_ylabel('Степень принадлежности μ', fontsize=14)
ax2.set_title('Функции принадлежности лингвистической переменной «Карьерная направленность»', fontsize=16)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1.05)
ax2.set_xticks(np.arange(0, 1.1, 0.1))
ax2.set_yticks(np.arange(0, 1.1, 0.2))
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper center')

plt.tight_layout()
plt.savefig('figure4_career.png', dpi=300, bbox_inches='tight')
print('✅ График 4 сохранён: figure4_career.png')

# Показываем оба графика
plt.show()