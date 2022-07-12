import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.uic import *

import pulp as p
from pulp import *

from sympy import Symbol
from sympy import *

def get_const(expr):
    lst_expr = expr.split(' ')
    if lst_expr[-2] == '+':
        return float(lst_expr[-1])
    else:
        return float('-' + lst_expr[-1])

def round_str(expr):
    expr_split = expr.split(' ')

    expr_lst = []
    for i in expr_split:
        expr_lst.append(i.split('*'))

    results_lst = []
    for j in range(len(expr_lst)):
        if len(expr_lst[j]) == 1:
            results_lst.append(expr_lst[j][0])
            continue
        results_lst.append(str(round(float(expr_lst[j][0]), 4)) + '*')
        results_lst.append(str(expr_lst[j][1]))

    results_lst[len(results_lst) - 1] = str(round(float(results_lst[len(results_lst) - 1]), 4))

    expr = ''.join(results_lst)

    return expr

class Operation(QMainWindow):

    def __init__(self):
        QMainWindow.__init__(self)
        loadUi("qt_design.ui", self)
        self.setWindowTitle("Med research")
        self.calculate.clicked.connect(self.show_formula)
    #

    def show_formula(self):
        u1 = Symbol('u1')
        u2 = Symbol('u2')
        u3 = Symbol('u3')
        u4 = Symbol('u4')
        model1 = self.model1.toPlainText()
        model2 = self.model2.toPlainText()
        model3 = self.model3.toPlainText()
        model4 = self.model4.toPlainText()
        model5 = self.model5.toPlainText()

        x1 = self.x1.value()
        x2 = self.x2.value()
        x3 = self.x3.value()
        x4 = self.x4.value()
        x5 = self.x5.value()
        x6 = self.x6.value()
        x7 = self.x7.value()
        x8 = self.x8.value()
        x9 = self.x9.value()
        x10 = self.x10.value()
        x11 = self.x11.value()
        x12 = self.x12.value()
        x13 = self.x13.value()
        x14 = self.x14.value()
        x15 = self.x15.value()
        x16 = self.x16.value()
        x17 = self.x17.value()
        x18 = self.x18.value()
        x19 = self.x19.value()
        x20 = self.x20.value()
        x21 = self.x21.value()
        x22 = self.x22.value()
        x23 = self.x23.value()

        subs_dict = {'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4,
                     'x5': x5, 'x6': x6, 'x7': x7, 'x8': x8,
                     'x9': x9, 'x10': x10, 'x11': x11, 'x12': x12,
                     'x13': x13, 'x14': x14, 'x15': x15, 'x16': x16,
                     'x17': x17, 'x18': x18, 'x19': x19, 'x20': x20,
                     'x21': x21, 'x22': x22, 'x23': x23}

        expr1 = sympify(model1).evalf(subs=subs_dict)
        expr2 = sympify(model2).evalf(subs=subs_dict)
        expr3 = sympify(model3).evalf(subs=subs_dict)
        expr4 = sympify(model4).evalf(subs=subs_dict)
        expr5 = sympify(model5).evalf(subs=subs_dict)

        self.m1.setText(round_str(str(expr1)))
        self.m2.setText(round_str(str(expr2)))
        self.m3.setText(round_str(str(expr3)))
        self.m4.setText(round_str(str(expr4)))
        self.m5.setText(round_str(str(expr5)))

        const1 = get_const(str(expr1))
        const2 = get_const(str(expr2))
        const3 = get_const(str(expr3))
        const4 = get_const(str(expr4))
        const5 = get_const(str(expr5))

        ar = str(expr2).split(' ')
        tr11 = round(float(ar[0].split('*')[0]), 4)
        tr12 = round(float(ar[2].split('*')[0]), 4)
        tr13 = round(float(ar[4].split('*')[0]), 4)
        tr14 = round(float(ar[6].split('*')[0]), 4)
        print(tr11, tr12, tr13, tr14)

        Lp_prob = LpProblem('AI_min_optimization', LpMinimize)

        u1 = LpVariable("u2", lowBound=self.min_u_1.value(), upBound=self.max_u_1.value())
        u2 = LpVariable("u3", lowBound=self.min_u_2.value(), upBound=self.max_u_2.value())
        u3 = LpVariable("u5", lowBound=self.min_u_3.value(), upBound=self.max_u_3.value())
        u4 = LpVariable("u6", lowBound=self.min_u_4.value(), upBound=self.max_u_4.value())

        Lp_prob += (tr11)*u1+(tr12)*u2-(tr13)*u3-(tr14)*u4+(const2)

        Lp_prob += LpConstraint(0.4749*u1-1.7105*u2+1.8998*u3-1.8089*u4+(const1), LpConstraintGE,
                                0), 'GA<='
        Lp_prob += LpConstraint(0.4749*u1-1.7105*u2+1.8998*u3-1.8089*u4+(const1)-10, LpConstraintLE,
                                10), 'GA>='

        Lp_prob += LpConstraint(18.6325*u1-25.6653*u2+22.7113*u3-11.6881*u4+(const3), LpConstraintGE,
                                0), 'PAG<='
        Lp_prob += LpConstraint(18.6325*u1-25.6653*u2+22.7113*u3-11.6881*u4+(const3)-10, LpConstraintLE,
                                10), 'PAG>='

        Lp_prob += LpConstraint(-5.4381*u1+5.8089*u2-2.1135*u3-1.6108*u4+(const4)-55, LpConstraintGE,
                                55), 'EF<='
        Lp_prob += LpConstraint(-5.4381*u1+5.8089*u2-2.1135*u3-1.6108*u4+(const4)-85, LpConstraintLE,
                                85), 'EF>='

        Lp_prob += LpConstraint(-2.8237*u1+7.9461*u2+0.9453*u3-10.7048*u4+(const5)-35, LpConstraintGE,
                                35), 'EDI<='
        Lp_prob += LpConstraint(-2.8237*u1+7.9461*u2+0.9453*u3-10.7048*u4+(const5)-85, LpConstraintLE,
                                85), 'EDI>='

        status = Lp_prob.solve()
        print(LpStatus[status])

        if LpStatus[status] == 'Infeasible':
            msg = QMessageBox()
            msg.setWindowTitle("Error")
            msg.setText("Infeasible")
            msg.setIcon(QMessageBox.Warning)

            msg.exec_()

        self.u_1.setText(str(round(p.value(u1), 4)))
        self.u_2.setText(str(round(p.value(u2), 4)))
        if p.value(u3) < 1.5:
            self.u_3.setText(str(round(p.value(u3), 4)+0.5))
        else:
            self.u_3.setText(str(round(p.value(u3), 4)))
        self.u_4.setText(str(round(p.value(u4), 4)))



        u1 = p.value(u1)
        u2 = p.value(u2)
        u3 = p.value(u3)
        u4 = p.value(u4)

        crit_dict = {'u1': p.value(u1), 'u2': p.value(u2), 'u3': p.value(u3), 'u4': p.value(u4)}

        GA = sympify(f'0.4749*u1-1.7105*u2+1.8998*u3-1.8089*u4+{(const1)}').evalf(subs=crit_dict)
        PAG = sympify(f'18.6325*u1-25.6653*u2+22.7113*u3-11.6881*u4+{(const3)}').evalf(subs=crit_dict)
        EF = sympify(f'-5.4381*u1+5.8089*u2-2.1135*u3-1.6108*u4+{(const4)}').evalf(subs=crit_dict)
        EDI = sympify(f'-2.8237*u1+7.9461*u2+0.9453*u3-10.7048*u4+{(const5)}').evalf(subs=crit_dict)

        print(round(GA, 4))

        self.prob_1.setText(str(round(GA, 4)))
        self.prob_2.setText(str(abs(round(p.value(Lp_prob.objective), 4))))
        self.prob_3.setText(str(round(PAG, 0)))
        self.prob_4.setText(str(round(EF, 0)))
        self.prob_5.setText(str(round(EDI, 0)))


if __name__ == '__main__':
    app = QApplication([])
    main_window = Operation()
    main_window.show()
    app.exec_()