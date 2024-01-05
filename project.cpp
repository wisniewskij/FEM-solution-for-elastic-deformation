#include <bits/stdc++.h>
#include "matplotlibcpp.h"
using namespace std;

// Two point Gaussian quadrature for given range [a, b]
double gaussian_quadrature(double a, double b, function<double(double)> funk) {
    if(a > b) swap(a, b);
    double point = 1 / sqrt(3), d1 = (b - a) / 2, d2 = (b + a) / 2;
    return d1 * (funk(-point * d1 + d2) + funk(point * d1 + d2));
}

// Integration function calling Gaussian quadrature on given domain ranges
double integrate(double a, double b, function<double(double)> funk, vector<pair<double, double>> &domain) {
    sort(domain.begin(), domain.end());
    double sum = 0.0, last = a;
    for(auto [c, d] : domain) {
        double seg_start = max(last, c), seg_end = min(d, b);
        if(seg_start >= seg_end) continue;
        sum += gaussian_quadrature(seg_start, seg_end, funk);
        last = d;
    }
    return sum;
}

// Linear equatino solver using gauss elimination method with partial pivoting
vector<double> solve_linear_equation_system(vector<vector<double>> &M) {
    vector<double> X(M.size());

    auto partial_pivot = [&]() {
        int n = M.size(); 
        for (int i = 0; i < n - 1; i++) {
            int pivot_row = i;

            for (int j = i+1; j < n; j++) if (abs(M[j][i]) > abs(M[pivot_row][i])) 
                pivot_row = j;
            
            if (pivot_row != i) for (int j = i; j <= n; j++) 
                swap(M[i][j], M[pivot_row][j]);
                
            for (int j = i+1; j < n; j++) {
                double factor = M[j][i] / M[i][i];
                for (int k = i; k <= n; k++) 
                    M[j][k] -= factor * M[i][k];
            }
        }
    };
    
    auto back_substitution = [&]() {
        int n = M.size();
        for (int i = n-1; i >= 0; i--) {
            double sum = 0;
            for (int j = i+1; j < n; j++) 
                sum += M[i][j] * X[j];
            
            X[i] = (M[i][n] - sum) / M[i][i];
        }
    };

    partial_pivot();
    back_substitution();
    return X;
}

// FEM solver 
vector<double> FEM(int n) {
    double a = 0.0, b = 2.0, h = (b-a)/((double)n);
    vector<function<double(double)>> e, de;
    vector<vector<pair<double, double>>> domain;
    for(int i=0; i<=n; i++) { 
        double lower_bound = (i-1) * h, upper_bound = (i+1) * h, mid_point = i * h;
        domain.push_back({{lower_bound, mid_point},{mid_point, upper_bound}});
        e.push_back([lower_bound, mid_point, upper_bound, h, a, b](double x) {
            if(x < max(lower_bound, a) || x > min(upper_bound, b)) return 0.0;
            return (x < mid_point ? (x - lower_bound) / h : (upper_bound - x) / h);
        });
        de.push_back([lower_bound, mid_point, upper_bound, h, a, b](double x) {
            if(x < max(lower_bound, a) || x > min(upper_bound, b)) return 0.0;
            return (x < mid_point ? 1.0 / h : -1.0 / h);
        });
    }

    auto E = [](double x) {
        if (x<0 || x > 2) return 0.0;
        if (x<=1) return 3.0;
        return 5.0;
    };

    auto B = [&E, &n](function<double(double)> u, function<double(double)> phi, function<double(double)> du, function<double(double)> dphi, vector<pair<double,double>> functinos_domains) {
        return -3.0 * u(0) * phi(0) + integrate(0.0, 2.0, [&du, &dphi, &E](double x){return E(x) * dphi(x) * du(x);}, functinos_domains);
    };  

    auto L = [](function<double(double)> phi) {
        return -30.0 * phi(0);
    };

    // Last row and column are omitted due to Dirichlet boundary condition u(2) = 0
    vector<vector<double>> M(n, vector<double>(n+1, 0.0)); 
    for(int i=0;i<n;i++) {
        for(int j=-1;j<=1;j++)
            if(i+j >= 0 && i+j < n){
                vector<pair<double, double>> function_domains = {};
                function_domains.insert(function_domains.end(), domain[i].begin(), domain[i].end());
                function_domains.insert(function_domains.end(), domain[i+j].begin(), domain[i+j].end());
                M[i][i+j] = B(e[i+j], e[i], de[i+j], de[i], function_domains); 
            }
        M[i][M[0].size()-1] = L(e[i]);
    }

    vector<double> ans = solve_linear_equation_system(M);
    
    return ans;
}
 
 
int main() {
    vector<double> ans = FEM(1000);
    // for(auto x:ans) cout<<x<<" ";
     
    return 0;
}