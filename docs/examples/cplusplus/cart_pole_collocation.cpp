//
// Created by justin beri on 5/2/21.
//

#include <casadi/casadi.hpp>
using namespace casadi;

double T = 2;
int N = 25;
double h_k = T/(N-1);
double l = 0.5;
double m1 = 2;
double m2 = 0.5;
double g0 = 9.81;
double d = 0.8;
double dmax = 2*d;
double umax = 100;
int M = 5;

// Cart-pole dynamics
MX f(const MX& x, const MX& u) {
    Slice all;
    MX q1 = x(0,all);
    MX q2 = x(1,all);
    MX q1dot = x(2,all);
    MX q2dot = x(3,all);
    MX q1ddot = ( l * m2 * MX::sin(q2) * MX::pow(q2dot,2) + u + m2 * g0 * MX::cos(q2) * MX::sin(q2) ) / ( m1 + m2 * (1 - MX::pow(MX::cos(q2),2)) );
    MX q2ddot = - ( l * m2 * MX::cos(q2) * MX::sin(q2) * MX::pow(q2dot,2) + u * MX::cos(q2) + (m1 + m2) * g0 * MX::sin(q2) ) / ( l * m1 + l * m2 * (1 - MX::pow(MX::cos(q2),2)) );
    return vertcat(q1dot, q2dot, q1ddot, q2ddot);
}

std::vector<double> linspace(double first, double last, int N) {
    std::vector<double> temp(N);
    linspace(temp, first, last);
    return temp;
}

void write_results(std::string file_path, std::vector<std::string> headers, std::vector<std::vector<double>> data) {
    std::ofstream my_file(file_path);
    std::string temp = ",";
    for (int i=0; i<headers.size(); ++i) {
        if (i == headers.size()-1) {
            temp = "\n";
        }
        my_file << headers.at(i) << temp;
    }

    for (int i=0; i<data.front().size(); ++i) {
        std::string temp = ",";
        for (int j=0; j<data.size(); ++j) {
            if (j == data.size()-1) {
                temp = "\n";
            }
            my_file << data.at(j).at(i) << temp;
        }
    }
}

int main(int argc, char *argv[])
{
    // Indexing slices
    Slice sall;
    Slice si0(0,N-1); // a(1:end-1) or a[:-1]
    Slice si1(1,N); // a(2:end) or a[1:]
    Slice s4(0,4);

    // Optimisation helper class
    auto opti = Opti();

    // Create the decision variables
    auto p = opti.variable(M,N);

    // The objective function
    auto J = h_k/2.0 * MX::sum2(p(4,si0)*p(4,si0) + p(4,si1)*p(4,si1));
    opti.minimize(J);

    // State x
    MX xk = p(s4,si0);
    MX xk1 = p(s4, si1);
    // Control u
    MX uk = p(4,si0);
    MX uk1 = p(4,si1);

    // Add the Hermite-Simpson collocation constraints
    auto fk = f(xk,uk); // dynamics at f(x(i), u(i))
    auto fk1 = f(xk1,uk1);
    auto uc = (uk + uk1)/2;
    auto xc = 1/2.0 * (xk + xk1) + h_k/8.0 * (fk - fk1); // mid-point between knots
    auto fc = f(xc,uc);
    auto G = xk-xk1 + h_k/6.0*(fk + 4.0*fc + fk1);
    opti.subject_to(G==0); // Add defect

    // Add path constraints
    opti.subject_to(-dmax<p(0,sall)<dmax);
    opti.subject_to(-umax<p(4,sall)<umax);

    // Add boundary constraints
    opti.subject_to(p(s4,0)==std::vector<double>{0, 0, 0, 0});
    opti.subject_to(p(s4,N-1)==std::vector<double>{d, M_PI, 0, 0});

    // Initial guess (linear interpolation)
    opti.set_initial(p(0,sall), linspace(0,d,N));
    opti.set_initial(p(1,sall), linspace(0,M_PI,N));
    opti.set_initial(p(2,sall), linspace(0,0,N));
    opti.set_initial(p(3,sall), linspace(0,0,N));
    opti.set_initial(p(4,sall), linspace(0,0,N));

    // Solve the nlp
    opti.solver("ipopt");
    auto sol = opti.solve();

    // Write out the results to a csv file
    auto t = linspace(0,T,N);
    write_results("out.csv", {"t","q1","q2","q1dot","q2dot","u"}, {t,
                                                                   std::vector<double>(sol.value(p(0,sall))),
                                                                   std::vector<double>(sol.value(p(1,sall))),
                                                                   std::vector<double>(sol.value(p(2,sall))),
                                                                   std::vector<double>(sol.value(p(3,sall))),
                                                                   std::vector<double>(sol.value(p(4,sall)))}
    );

    return 0;
}