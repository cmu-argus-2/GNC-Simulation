
// This is effectively a wrapper for exprtk.h so it is only compiled once (in
// this translation unit). Compiling exprtk takes a long time.

#include "exprtk.hpp"

inline double deg_2_rad(double x) {
    // purposely not using DEG_2_RAD macro from utils_and_transforms.h because
    // then every time I changed utils_and_transforms.h, this source file would
    // need to be recompiled, and it takes forever to compile
    return x * (M_PI / 180.0);
}

double evaluate(const std::string& expression_string) {
    exprtk::symbol_table<double> symbol_table;
    symbol_table.add_function("DEG_2_RAD", deg_2_rad);
    symbol_table.add_constant("HR_PER_SEC", 1 / 3600.0);
    symbol_table.add_constants();

    exprtk::expression<double> expression;
    expression.register_symbol_table(symbol_table);

    exprtk::parser<double> parser;
    parser.compile(expression_string, expression);
    return expression.value();
}