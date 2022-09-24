use proc_macro::TokenStream;

#[proc_macro_attribute]
pub fn select(_attr: TokenStream, input: TokenStream) -> TokenStream {
    input
}

#[proc_macro_attribute]
pub fn generic(_attr: TokenStream, input: TokenStream) -> TokenStream {
    input
}
