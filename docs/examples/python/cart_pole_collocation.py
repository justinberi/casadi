import casadi as ca
import matplotlib.pyplot as plt

T = 4;
N = 25;
h_k = T/(N-1);
l = 0.5;
m1 = 2;
m2 = 0.5;
g = 9.81;
d = 0.8;
dmax = 2*d;
umax = 100;
M = 5;

# q1 = ca.SX.sym('q1')
# q2 = ca.SX.sym('q2')
# q1dot = ca.SX.sym('q1dot')
# q2dot = ca.SX.sym('q2dot')
# u = ca.SX.sym('u')
# x = ca.vertcat(q1,q2,q1dot,q2dot)
# q1ddot = ( l * m2 * ca.sin(q2) * ca.power(q2dot,2) + u + m2 * g * ca.cos(q2) * ca.sin(q2) ) / ( m1 + m2 * (1 - ca.power(ca.cos(q2),2)) );
# q2ddot = - ( l * m2 * ca.cos(q2) * ca.sin(q2) * ca.power(q2dot,2) + u * ca.cos(q2) + (m1 + m2) * g * ca.sin(q2) ) / ( l * m1 + l * m2 * (1 - ca.power(ca.cos(q2),2)) );
# f = ca.Function('f',[x,u],[ca.vertcat(q1dot,q2dot,q1ddot,q2ddot)])

def f(x,u):
    q1 = x[0,:]
    q2 = x[1,:]
    q1dot = x[2,:]
    q2dot = x[3,:]
    q1ddot = ( l * m2 * ca.sin(q2) * ca.power(q2dot,2) + u + m2 * g * ca.cos(q2) * ca.sin(q2) ) / ( m1 + m2 * (1 - ca.power(ca.cos(q2),2)) );
    q2ddot = - ( l * m2 * ca.cos(q2) * ca.sin(q2) * ca.power(q2dot,2) + u * ca.cos(q2) + (m1 + m2) * g * ca.sin(q2) ) / ( l * m1 + l * m2 * (1 - ca.power(ca.cos(q2),2)) );
    return ca.vertcat(q1dot,q2dot,q1ddot,q2ddot)


# Set up the optimisation problem
opti = ca.Opti();

p = opti.variable(M,N);

# Objective function
#J = h_k/2*ca.sum2(ca.power(p[4,:-1],2) + ca.power(p[4,1:],2))
J = h_k/2*ca.sum2(ca.power(p[4,:-1],2) + ca.power(p[4,1:],2))
opti.minimize(J)

# Interpolation constraint symbolic expressions
xk = p[:4,:-1]
xk1 = p[:4,1:]
uk = p[4,:-1]
uk1 = p[4,1:]
fk = f(xk,uk)
fk1 = f(xk1,uk1)
uc = (uk + uk1)/2
xc = 1/2 * (xk + xk1) + h_k/8 * (fk - fk1)
fc = f(xc,uc)
G = xk-xk1 + h_k/6*(fk + 4*fc + fk1)
opti.subject_to(G==0) # Add interpolation constraint


# Add path constraints
opti.subject_to(p[0,:]<dmax)
opti.subject_to(-dmax<p[0,:])
opti.subject_to(p[4,:]<umax)
opti.subject_to(-umax<p[4,:])

# Add boundary constraints
opti.subject_to(p[:4,0]==[0,0,0,0])
opti.subject_to(p[:4,-1]==[d,ca.pi,0,0])

# Intialise
opti.set_initial(p[0,:],ca.linspace(0,d,N));
opti.set_initial(p[1,:],ca.linspace(0,ca.pi,N));
opti.set_initial(p[2,:],ca.linspace(0,0,N));
opti.set_initial(p[3,:],ca.linspace(0,0,N));
opti.set_initial(p[4,:],ca.linspace(0,0,N));

opti.solver('ipopt');
sol = opti.solve();

# Animate the stuff
x = sol.value(p[0,:]);
theta = sol.value(p[1,:])
t = ca.linspace(0,T,N)
plt.plot(t,x)
plt.show()