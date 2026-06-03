# Создай файл plot_budget.py в корне проекта:
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 300000, 1000)

low = np.where(x <= 30000, np.where(x <= 0, 0, x/30000 if 30000!=0 else 0), 
       np.where(x <= 60000, 1 - (x-30000)/30000, 0))
low = np.clip(np.where(x <= 30000, x/30000, np.where(x <= 60000, 1 - (x-30000)/30000, 0)), 0, 1)

medium = np.where(x < 20000, 0, np.where(x <= 50000, (x-20000)/30000, 
          np.where(x <= 100000, 1 - (x-50000)/50000, 0)))

high = np.where(x <= 80000, 0, np.where(x <= 150000, (x-80000)/70000, 1))

plt.figure(figsize=(10, 6))
plt.plot(x/1000, low, 'b-', linewidth=2, label='Низкий')
plt.plot(x/1000, medium, 'g-', linewidth=2, label='Средний')
plt.plot(x/1000, high, 'r-', linewidth=2, label='Высокий')
plt.xlabel('Бюджет (тыс. руб.)')
plt.ylabel('Степень принадлежности μ(x)')
plt.title('Рисунок 1 – Функции принадлежности переменной «Бюджет»')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figure1_budget.png', dpi=300)
print('График сохранён: figure1_budget.png')