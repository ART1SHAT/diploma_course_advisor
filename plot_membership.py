import numpy as np
import matplotlib.pyplot as plt

# Данные для графика (из вашего диплома)
budget_range = np.linspace(0, 150000, 500)

# Функции принадлежности (трапецеидальные/треугольные)
# Низкий бюджет: [0, 0, 30000, 60000]
low = np.where(budget_range <= 30000, 
               np.where(budget_range <= 0, 0, 
                       np.where(budget_range <= 30000, budget_range/30000,
                               np.where(budget_range <= 60000, 1 - (budget_range-30000)/30000, 0))),
               0)
low = np.clip(np.where(budget_range <= 30000, budget_range/30000, 
                       np.where(budget_range <= 60000, 1 - (budget_range-30000)/30000, 0)), 0, 1)

# Средний бюджет: [40000, 70000, 100000] (треугольная)
medium = np.where(budget_range < 40000, 0,
                  np.where(budget_range <= 70000, (budget_range-40000)/30000,
                          np.where(budget_range <= 100000, 1 - (budget_range-70000)/30000, 0)))

# Высокий бюджет: [80000, 150000, 150000, 150000]
high = np.where(budget_range <= 80000, 0,
                np.where(budget_range <= 150000, (budget_range-80000)/70000, 1))

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(budget_range/1000, low, linewidth=2.5, label='Низкий', color='#2563eb')
plt.plot(budget_range/1000, medium, linewidth=2.5, label='Средний', color='#16a34a')
plt.plot(budget_range/1000, high, linewidth=2.5, label='Высокий', color='#dc2626')

plt.xlabel('Бюджет (тыс. ₽)', fontsize=12)
plt.ylabel('Степень принадлежности μ(x)', fontsize=12)
plt.title('Функции принадлежности переменной «Бюджет»', fontsize=14, fontweight='bold')
plt.legend(loc='upper right', fontsize=11)
plt.grid(True, alpha=0.3, linestyle='--')
plt.xlim(0, 150)
plt.ylim(0, 1.1)
plt.xticks(range(0, 151, 20))
plt.yticks(np.arange(0, 1.1, 0.2))

plt.tight_layout()
plt.savefig('budget_membership.png', dpi=300, bbox_inches='tight')
plt.show()