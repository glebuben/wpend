# Основные формулы — колёсный маятник (Segway)

Состояние: $x = (\theta,\ \varphi,\ \dot\theta,\ \dot\varphi)$, где $\theta$ — наклон корпуса от вертикали ($0$ — вертикаль), $\varphi$ — угол поворота колеса в мировой системе (качение без проскальзывания).

Лумпированные параметры:

$$
\alpha = I_b + m_b l^2,\qquad
\beta = m_b r l,\qquad
\gamma = I_w + (m_w + m_b) r^2,\qquad
D = m_b g l,
$$

$$
b(\theta) = \gamma + \beta\cos\theta,\qquad
\Delta(\theta) = \alpha\gamma - \beta^2\cos^2\theta > 0,\qquad
\Delta_0 = \Delta(0) = \alpha\gamma - \beta^2.
$$

---

## 1. Динамика $f(s)$

### 1.1 Полная модель (Эйлер–Лагранж)

С обобщёнными координатами $q=(\theta,\varphi)^\top$ и моментом мотора $\tau=(u,-u)^\top$:

$$
M(\theta)\,\ddot q + C(\theta,\dot\theta)\,\dot q + G(\theta) = \tau,
$$

$$
M = \begin{pmatrix}\alpha & \beta\cos\theta \\ \beta\cos\theta & \gamma\end{pmatrix},
\qquad
G = \begin{pmatrix}-D\sin\theta \\ 0\end{pmatrix},
\qquad
C\dot q = \begin{pmatrix}0 \\ -\beta\sin\theta\,\dot\theta^2\end{pmatrix}.
$$

Покомпонентно:

$$
\alpha\,\ddot\theta + \beta\cos\theta\,\ddot\varphi - D\sin\theta = u,
$$

$$
\beta\cos\theta\,\ddot\theta + \gamma\,\ddot\varphi - \beta\sin\theta\,\dot\theta^2 = -u.
$$

### 1.2 Приведённая динамика наклона (2-D в $(\theta,\dot\theta)$)

После исключения $\ddot\varphi$ (координата $\varphi$ циклическая):

$$
\boxed{\;\ddot\theta = \frac{b(\theta)\,u + \gamma D\sin\theta - \beta^2\sin\theta\cos\theta\,\dot\theta^2}{\Delta(\theta)}\;}
$$

Ни $\varphi$, ни $\dot\varphi$ не входят (галилеева инвариантность идеального качения без проскальзывания).

### 1.3 Векторное поле $f(s)$ в координатах $(\theta,\varphi)$

Решая систему §1.1 относительно ускорений (по Крамеру, определитель $=\Delta(\theta)$), поле $\dot s = f(s,u)$ для $s=(\theta,\varphi,\dot\theta,\dot\varphi)$:

$$
f(s,u) =
\begin{pmatrix}
\dot\theta \\[2pt]
\dot\varphi \\[2pt]
\dfrac{b(\theta)\,u + \gamma D\sin\theta - \beta^2\sin\theta\cos\theta\,\dot\theta^2}{\Delta(\theta)} \\[10pt]
\ddot\varphi
\end{pmatrix},
$$

где $\ddot\varphi$ в явном виде:

$$
\ddot\varphi = \frac{\alpha\beta\sin\theta\,\dot\theta^2 - \beta D\sin\theta\cos\theta - (\alpha+\beta\cos\theta)\,u}{\Delta(\theta)}.
$$

Ни $\varphi$, ни $\dot\varphi$ не входят в $\ddot\theta$ (координата $\varphi$ циклическая; см. §1.4).

### 1.4 Почему $\ddot\theta$ не зависит от $\dot\varphi$ (сокращение в лагранжиане)

Кинетическая и потенциальная энергии:

$$
T = \tfrac12\,\alpha\,\dot\theta^2 + \beta\cos\theta\,\dot\varphi\,\dot\theta + \tfrac12\,\gamma\,\dot\varphi^2,
\qquad
V = m_b g\,(r + l\cos\theta).
$$

$\dot\varphi$ связывается с наклоном только через перекрёстный член $\beta\cos\theta\,\dot\varphi\dot\theta$. В уравнении Эйлера–Лагранжа по $\theta$ этот член даёт два вклада, которые уничтожаются:

$$
\frac{d}{dt}\frac{\partial T}{\partial\dot\theta}
= \alpha\ddot\theta + \beta\cos\theta\,\ddot\varphi \underbrace{-\,\beta\sin\theta\,\dot\theta\dot\varphi}_{\text{из }\frac{d}{dt}\cos\theta},
\qquad
\frac{\partial T}{\partial\theta} = \underbrace{-\,\beta\sin\theta\,\dot\varphi\dot\theta}_{\text{тот же член}},
$$

$$
\frac{d}{dt}\frac{\partial T}{\partial\dot\theta} - \frac{\partial T}{\partial\theta}
= \alpha\ddot\theta + \beta\cos\theta\,\ddot\varphi
\;\underbrace{-\,\beta\sin\theta\,\dot\theta\dot\varphi + \beta\sin\theta\,\dot\varphi\dot\theta}_{=\,0}.
$$

Остаётся $\alpha\ddot\theta + \beta\cos\theta\,\ddot\varphi - D\sin\theta = u$ — без $\dot\varphi$. Сокращение неслучайно: коэффициент перекрёстного члена $\beta\cos\theta$ зависит только от $\theta$, поэтому «центробежный» вклад от $\tfrac{d}{dt}$ и «потенциальный» от $-\partial/\partial\theta$ равны и противоположны. После исключения $\ddot\varphi$ (уравнение колеса тоже $\dot\varphi$-свободно) $\ddot\theta$ остаётся функцией только $(\theta,\dot\theta,u)$.

Физически это галилеева инвариантность идеальной модели: качение без трения не «чувствует» абсолютную скорость катания. Корпус реагирует на *ускорение* колеса $\ddot\varphi$ (через инерционную связь), но не на его скорость $\dot\varphi$ — как маятник в вагоне, который отклоняет ускорение, а не постоянная скорость.

### 1.5 Что возвращает зависимость от $\dot\varphi$

Любая скоростно-зависимая сила (трение качения, задняя-ЭДС мотора, аэродинамическое сопротивление) добавляет в уравнение колеса член $\propto\dot\varphi$. Например, вязкий момент $-c\,\dot\varphi$:

$$
\beta\cos\theta\,\ddot\theta + \gamma\,\ddot\varphi - \beta\sin\theta\,\dot\theta^2 = -u - c\,\dot\varphi.
$$

Исключение $\ddot\varphi$ теперь оставляет $\dot\varphi$ в динамике наклона:

$$
\ddot\theta = \frac{b(\theta)\,u + \gamma D\sin\theta - \beta^2\sin\theta\cos\theta\,\dot\theta^2 + \beta c\cos\theta\,\dot\varphi}{\Delta(\theta)}.
$$

Слагаемое $\beta c\cos\theta\,\dot\varphi/\Delta$ рушит галилееву инвариантность, и задача «упадёт ли корпус» становится честно 3-мерной в $(\theta,\dot\theta,\dot\varphi)$ (граница восстановимости — поверхность, а не кривая; ср. `CLAUDE.md` §6.7).

---

## 2. Якобиан системы

Поле $f(s,u)$ полностью:

$$
f =
\begin{pmatrix}
\dot\theta \\
\dot\varphi \\
\ddot\theta(\theta,\dot\theta,u) \\
\ddot\varphi(\theta,\dot\theta,u)
\end{pmatrix},
\qquad
\ddot\theta = \frac{N(\theta,\dot\theta,u)}{\Delta},\quad
\ddot\varphi = \frac{R(\theta,\dot\theta,u)}{\Delta},
$$

$$
N = b\,u + \gamma D\sin\theta - \beta^2\sin\theta\cos\theta\,\dot\theta^2,
\qquad
R = \alpha\beta\sin\theta\,\dot\theta^2 - \beta D\sin\theta\cos\theta - (\alpha+\beta\cos\theta)\,u.
$$

Якобиан $J = \partial f/\partial s$, $s=(\theta,\varphi,\dot\theta,\dot\varphi)$:

$$
J =
\begin{pmatrix}
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1 \\
\partial_\theta\ddot\theta & 0 & \partial_{\dot\theta}\ddot\theta & 0 \\
\partial_\theta\ddot\varphi & 0 & \partial_{\dot\theta}\ddot\varphi & 0
\end{pmatrix}.
$$

Столбцы $\varphi$ и $\dot\varphi$ нулевые — прямое следствие $\dot\varphi$-независимости из §1.4. Ненулевые элементы (с $\Delta' = \beta^2\sin 2\theta$):

$$
\partial_{\dot\theta}\ddot\theta = -\frac{\beta^2\sin 2\theta\,\dot\theta}{\Delta},
\qquad
\partial_{\dot\theta}\ddot\varphi = \frac{2\alpha\beta\sin\theta\,\dot\theta}{\Delta},
$$

$$
\partial_\theta\ddot\theta = \frac{N_\theta\,\Delta - N\,\Delta'}{\Delta^2},
\qquad
N_\theta = -\beta\sin\theta\,u + \gamma D\cos\theta - \beta^2\cos 2\theta\,\dot\theta^2,
$$

$$
\partial_\theta\ddot\varphi = \frac{R_\theta\,\Delta - R\,\Delta'}{\Delta^2},
\qquad
R_\theta = \alpha\beta\cos\theta\,\dot\theta^2 - \beta D\cos 2\theta + \beta\sin\theta\,u.
$$

### 2.1 Полностью раскрытые элементы $\partial_\theta$

Подставляя $N,\ N_\theta,\ R,\ R_\theta,\ \Delta=\alpha\gamma-\beta^2\cos^2\theta,\ \Delta'=\beta^2\sin 2\theta$ в один дробный вид:

$$
\begin{aligned}
\partial_\theta\ddot\theta = \frac{1}{\big(\alpha\gamma-\beta^2\cos^2\theta\big)^2}\Big[
&\big(-\beta\sin\theta\,u + \gamma D\cos\theta - \beta^2\cos 2\theta\,\dot\theta^2\big)\big(\alpha\gamma-\beta^2\cos^2\theta\big) \\
&- \beta^2\sin 2\theta\,\big((\gamma+\beta\cos\theta)u + \gamma D\sin\theta - \beta^2\sin\theta\cos\theta\,\dot\theta^2\big)\Big],
\end{aligned}
$$

$$
\begin{aligned}
\partial_\theta\ddot\varphi = \frac{1}{\big(\alpha\gamma-\beta^2\cos^2\theta\big)^2}\Big[
&\big(\alpha\beta\cos\theta\,\dot\theta^2 - \beta D\cos 2\theta + \beta\sin\theta\,u\big)\big(\alpha\gamma-\beta^2\cos^2\theta\big) \\
&- \beta^2\sin 2\theta\,\big(\alpha\beta\sin\theta\,\dot\theta^2 - \beta D\sin\theta\cos\theta - (\alpha+\beta\cos\theta)u\big)\Big].
\end{aligned}
$$

Элементы по $\dot\theta$ уже в конечном виде:

$$
\partial_{\dot\theta}\ddot\theta = -\frac{\beta^2\sin 2\theta\,\dot\theta}{\alpha\gamma-\beta^2\cos^2\theta},
\qquad
\partial_{\dot\theta}\ddot\varphi = \frac{2\alpha\beta\sin\theta\,\dot\theta}{\alpha\gamma-\beta^2\cos^2\theta}.
$$

### 2.2 Линеаризация в верхнем положении

В равновесии $s=0,\ u=0$ (с $\Delta_0 = \alpha\gamma - \beta^2$):

$$
A = J\big|_0 =
\begin{pmatrix}
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1 \\
\dfrac{\gamma D}{\Delta_0} & 0 & 0 & 0 \\[6pt]
-\dfrac{\beta D}{\Delta_0} & 0 & 0 & 0
\end{pmatrix},
\qquad
B = \frac{\partial f}{\partial u}\bigg|_0 =
\begin{pmatrix}
0 \\ 0 \\ \dfrac{\gamma+\beta}{\Delta_0} \\[6pt] -\dfrac{\alpha+\beta}{\Delta_0}
\end{pmatrix}.
$$

Наклонная физика $A[2,0]=\gamma D/\Delta_0$ не зависит от выбора координат колеса (ср. `CLAUDE.md` §9.7; проверка `scripts/verify/verify_lqr.py`).

### 2.3 Гессиан

Гессиан поля — набор матриц вторых производных $H_i = \partial^2 f_i/\partial s^2$, $i=1..4$. Компоненты $f_1=\dot\theta,\ f_2=\dot\varphi$ линейны, поэтому $H_1=H_2=0$. Компоненты $f_3=\ddot\theta,\ f_4=\ddot\varphi$ зависят только от $(\theta,\dot\theta)$, так что у каждой ненулевой лишь блок $2\times2$ в координатах $(\theta,\dot\theta)$ (строки/столбцы $\varphi,\dot\varphi$ — нули):

$$
H_{\ddot\theta} =
\begin{pmatrix}
\partial^2_{\theta\theta}\ddot\theta & \partial^2_{\theta\dot\theta}\ddot\theta \\[2pt]
\partial^2_{\theta\dot\theta}\ddot\theta & \partial^2_{\dot\theta\dot\theta}\ddot\theta
\end{pmatrix},
\qquad
H_{\ddot\varphi} =
\begin{pmatrix}
\partial^2_{\theta\theta}\ddot\varphi & \partial^2_{\theta\dot\theta}\ddot\varphi \\[2pt]
\partial^2_{\theta\dot\theta}\ddot\varphi & \partial^2_{\dot\theta\dot\theta}\ddot\varphi
\end{pmatrix}.
$$

Для $g\in\{\ddot\theta,\ddot\varphi\}$ с числителем $Q\in\{N,R\}$ (тот же формат, что у якобиана):

$$
\partial^2_{\theta\theta}g = \frac{Q_{\theta\theta}}{\Delta} - \frac{2Q_\theta\,\Delta' + Q\,\Delta''}{\Delta^2} + \frac{2Q\,(\Delta')^2}{\Delta^3},
$$

$$
\partial^2_{\theta\dot\theta}g = \frac{Q_{\theta\dot\theta}\,\Delta - Q_{\dot\theta}\,\Delta'}{\Delta^2},
\qquad
\partial^2_{\dot\theta\dot\theta}g = \frac{Q_{\dot\theta\dot\theta}}{\Delta}.
$$

Строительные блоки ($\Delta' = \beta^2\sin 2\theta,\ \Delta'' = 2\beta^2\cos 2\theta$), для $Q=N$:

$$
N_{\theta\theta} = -\beta\cos\theta\,u - \gamma D\sin\theta + 2\beta^2\sin 2\theta\,\dot\theta^2,
\qquad
N_{\theta\dot\theta} = -2\beta^2\cos 2\theta\,\dot\theta,
\qquad
N_{\dot\theta\dot\theta} = -\beta^2\sin 2\theta,
$$

для $Q=R$:

$$
R_{\theta\theta} = -\alpha\beta\sin\theta\,\dot\theta^2 + 2\beta D\sin 2\theta + \beta\cos\theta\,u,
$$

$$
R_{\theta\dot\theta} = 2\alpha\beta\cos\theta\,\dot\theta,
\qquad
R_{\dot\theta\dot\theta} = 2\alpha\beta\sin\theta.
$$

(значения $N, N_\theta, R, R_\theta$ — из §2.) В верхнем положении $s=0,\ u=0$ каждый блок пропорционален $\sin\theta$, $\dot\theta$ или их произведению и $\Delta'=0$, поэтому

$$
H_{\ddot\theta}\big|_0 = H_{\ddot\varphi}\big|_0 = 0.
$$

То есть у динамики наклона нет квадратичного члена в вертикали — ведущая нелинейность кубическая по $(\theta,\dot\theta)$, чем и объясняется относительно большая точность LQR-линеаризации (ср. `CLAUDE.md` §9.2; карты нормы гессиана — `src/hessian.py`, `scripts/figures/hessian_norms.py`).

---

## 3. Первые интегралы $H_\pm$ (постоянный момент)

При постоянном $u$ приведённая динамика наклона допускает энергетический первый интеграл:

$$
\boxed{\;H(\theta,\dot\theta;u) = \tfrac12\,\Delta(\theta)\,\dot\theta^2 - u\,(\gamma\theta + \beta\sin\theta) + \gamma D\cos\theta\;}
$$

свойство: $\dot H = 0$ вдоль траекторий при постоянном $u$.

Для предельных допустимых моментов $u=\mp u_{\max}$:

$$
H_-(\theta,\dot\theta) = \tfrac12\,\Delta(\theta)\,\dot\theta^2 + u_{\max}(\gamma\theta + \beta\sin\theta) + \gamma D\cos\theta
\quad(u=-u_{\max},\ \text{тормоз вперёд}),
$$

$$
H_+(\theta,\dot\theta) = \tfrac12\,\Delta(\theta)\,\dot\theta^2 - u_{\max}(\gamma\theta + \beta\sin\theta) + \gamma D\cos\theta
\quad(u=+u_{\max},\ \text{тормоз назад}).
$$

### 3.1 Седло и его уровень

Седловое равновесие потока $u=-u_{\max}$ в точке $(\theta_{eq},0)$, $\theta_{eq}\in[0,\pi/2]$:

$$
u_{\max}\,(\gamma + \beta\cos\theta_{eq}) = \gamma D\sin\theta_{eq}
$$

(единственный корень в $(0,\pi/2)$ при $u_{\max}<D$; иначе $\theta_{eq}=\pi/2$). Уровень седла (сепаратриса):

$$
K(u_{\max}) = u_{\max}(\gamma\theta_{eq} + \beta\sin\theta_{eq}) + \gamma D\cos\theta_{eq}.
$$

### 3.2 Граница восстановимого множества

Состояние восстановимо $\iff H_+ < K$ и $H_- < K$, что через $\max(a+b,a-b)=a+|b|$ сводится к одному неравенству:

$$
\boxed{\;\tfrac12\,\Delta(\theta)\,\dot\theta^2 + u_{\max}\,|\gamma\theta + \beta\sin\theta| + \gamma D\cos\theta < K(u_{\max})\;}
$$

Граница — равенство; решая относительно $\dot\theta$:

$$
\dot\theta^2(\theta;u_{\max}) = \frac{2\big[\,K(u_{\max}) - u_{\max}\,|\gamma\theta + \beta\sin\theta| - \gamma D\cos\theta\,\big]}{\Delta(\theta)}.
$$

---

## Приложение / ссылки — выводы

Полные пошаговые выводы вынесены сюда, чтобы не загромождать основные формулы выше.

### A. Вывод приведённой динамики наклона $\ddot\theta$ (§1.2)

Из системы §1.1 второе уравнение даёт $\ddot\varphi = \big(-u + \beta\sin\theta\,\dot\theta^2 - \beta\cos\theta\,\ddot\theta\big)/\gamma$. Подстановка в первое уравнение и группировка по $\ddot\theta$ приводит множитель $\alpha - \beta^2\cos^2\theta/\gamma = \Delta(\theta)/\gamma$; домножение на $\gamma$ даёт формулу в рамке §1.2. Подробности — `docs/Segway_Lyapunov_Controller.docx`, §2, и символьная проверка `scripts/verify/verify_segway.py` (остатки $=0$).

### B. Вывод первого интеграла $H$ (§3)

Приведённое ОДУ при постоянном $u$:

$$
\Delta(\theta)\,\ddot\theta = b(\theta)u + \gamma D\sin\theta - \beta^2\sin\theta\cos\theta\,\dot\theta^2.\tag{$*$}
$$

Интегрирующий множитель — скорость $\dot\theta$. Домножая $(*)$ на $\dot\theta$ и используя

$$
\frac{d}{dt}\!\left[\tfrac12\Delta(\theta)\dot\theta^2\right] = \Delta\dot\theta\ddot\theta + \tfrac12\Delta'(\theta)\dot\theta^3,
\qquad \Delta'(\theta) = 2\beta^2\sin\theta\cos\theta,
$$

центробежные кубические члены $\beta^2\sin\theta\cos\theta\,\dot\theta^3$ сокращаются ровно. Остаётся

$$
\frac{d}{dt}\!\left[\tfrac12\Delta\dot\theta^2\right] = b\,u\,\dot\theta + \gamma D\sin\theta\,\dot\theta.
$$

Оба члена справа — полные производные (при постоянном $u$):

$$
b\,u\,\dot\theta = \frac{d}{dt}\big[u(\gamma\theta + \beta\sin\theta)\big],\qquad
\gamma D\sin\theta\,\dot\theta = -\frac{d}{dt}\big[\gamma D\cos\theta\big],
$$

откуда $\dot H = 0$ для $H$ из рамки §3. Смысл слагаемых: $\tfrac12\Delta\dot\theta^2$ — кинетическое, $\gamma D\cos\theta$ — гравитационный потенциал, $-u(\gamma\theta+\beta\sin\theta)$ — работа постоянного управляющего момента (его $-d/d\theta$ возвращает $b(\theta)u$). Полный вывод — `CLAUDE.md` §6.9; символьно/численно — `scripts/figures/recoverable_set_symbolic.py`, `scripts/verify/verify_recoverable_set.py`.

### C. Почему граница — это $H=K$ (§3.2)

$\dot H=0$ верно на всех уровнях при фиксированном $u$; границу выделяет именно тот уровень, что проходит через седло: $H=K$. Этот уровень — устойчивое многообразие седла, разделяющее ограниченные (восстановимые) и убегающие орбиты. При переменном $u\in[-u_{\max},u_{\max}]$: $\dot H_- = b(\theta)\dot\theta\,(u+u_{\max})$, и при $\dot\theta>0$ минимум по $u$ достигается на $u=-u_{\max}$ и равен нулю — максимальный тормоз лишь удерживает $H_-$. Обсуждение — `CLAUDE.md` §6.5.

### D. Первоисточник

Aguilar-Ibáñez, Gutiérrez Frias, Suárez Castañón, "Lyapunov-Based Controller for the Inverted Pendulum Cart System," *Nonlinear Dynamics* **40**, 367–374 (2005). PDF: `docs/Cart-pole controller.pdf`.
