
// This is effectively a wrapper for exprtk.h so it is only compiled once (in
// this translation unit). Compiling exprtk takes a long time.

/* 

    This is a wrapper for exprtk.hpp. If we didn't have ExpressionEvaluation.cpp, we would have had to include exprtk.hpp in the parameter parser. 
    The problem with that is it takes several minutes for the compiler to compile code from exprt.hpp, so each time we made a change to the parameter parser (such as including a new field to the param file), you'd have to wait several minutes for your code to build.

    By wrapping exprtk.hpp in the evaluate() function in ExpressionEvaluator.cpp, the code from expert.hpp only gets compiled once - when ExpressionEvaluator gets built. Then we link ExpressionEvaluator.o with the rest of the code.

    The deg_2_rad function might be causing your confusion with math/conversions.cpp. The expression evaluation code calls for a deg_2_rad function, whereas conversions.h only provides a macro. 
    But even if conversions.h had such a function, including conversions.h into ExpressionEvaluation.cpp would mean recompiling ExpressionEvalaution.cpp each time conversions.h changed, which would be very annoying.

    I agree code duplication is not ideal, but in this case, the benefits significantly outweigh the drawbacks.

*/

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