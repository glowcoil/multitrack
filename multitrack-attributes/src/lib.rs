use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    parse_macro_input, spanned::Spanned, Error, FnArg, GenericParam, ItemFn, Pat, TypeParamBound,
};

#[proc_macro_attribute]
pub fn specialize(_attr: TokenStream, input: TokenStream) -> TokenStream {
    let func = parse_macro_input!(input as ItemFn);

    let attrs = &func.attrs;
    let vis = &func.vis;
    let sig = &func.sig;

    let block = &func.block;

    let mut arch_ident = None;
    'arch: for param in &sig.generics.params {
        if let GenericParam::Type(ty) = param {
            for bound in &ty.bounds {
                if let TypeParamBound::Trait(tr) = bound {
                    if let Some(last) = tr.path.segments.last() {
                        if last.ident.to_string() == "Arch" {
                            arch_ident = Some(&ty.ident);
                            break 'arch;
                        }
                    }
                }
            }
        }
    }
    if arch_ident.is_none() {
        return Error::new(func.sig.generics.span(), "could not find Arch parameter")
            .into_compile_error()
            .into();
    }
    let arch_ident = arch_ident.unwrap();

    let mut generic_params = Vec::new();
    let mut generic_idents = Vec::new();
    for param in &sig.generics.params {
        generic_params.push(param);
        match param {
            GenericParam::Type(ty) => {
                generic_idents.push(&ty.ident);
            }
            GenericParam::Const(_) => {
                return Error::new(func.sig.generics.span(), "const generics not supported")
                    .into_compile_error()
                    .into();
            }
            GenericParam::Lifetime(_) => {}
        };
    }

    let mut inner_sig = sig.clone();
    inner_sig.ident = format_ident!("__inner");

    let mut arg_types = Vec::with_capacity(sig.inputs.len());
    let mut arg_fields = Vec::with_capacity(sig.inputs.len());
    let mut arg_names = Vec::with_capacity(sig.inputs.len());
    for (i, arg) in sig.inputs.iter().enumerate() {
        arg_types.push(format_ident!("I{}", i));
        arg_fields.push(format_ident!("_{}", i));

        match arg {
            FnArg::Receiver(_) => {
                return Error::new(func.span(), "associated functions not supported")
                    .into_compile_error()
                    .into();
            }
            FnArg::Typed(typed) => {
                if let Pat::Ident(ident) = &*typed.pat {
                    arg_names.push(ident);
                } else {
                    return Error::new(typed.span(), "argument patterns not supported")
                        .into_compile_error()
                        .into();
                }
            }
        }
    }

    let result = quote! {
        #(#attrs)*
        #vis #sig {
            struct __Task<#(#generic_idents,)* F, #(#arg_types,)*> {
                #(#arg_fields: ::core::mem::ManuallyDrop<#arg_types>,)*
                _f: F,
                _phantom: ::core::marker::PhantomData<(#(#generic_idents,)*)>,
            }

            impl<#(#generic_params,)* F, #(#arg_types,)* O> ::multitrack::Task for __Task<#(#generic_idents,)* F, #(#arg_types),*>
            where
                F: Fn(#(#arg_types),*) -> O,
            {
                type Result = O;

                #[inline(always)]
                fn run<__A: Arch>(self) -> O {
                    use ::core::mem::{ManuallyDrop, transmute_copy};

                    unsafe {
                        let res = __inner::<#(#generic_idents,)*>(
                            #(transmute_copy::<ManuallyDrop<#arg_types>, _>(&self.#arg_fields),)*
                        );
                        transmute_copy::<_, O>(&ManuallyDrop::new(res))
                    }
                }
            }

            #[inline(always)]
            #inner_sig #block

            #arch_ident::invoke(__Task {
                #(#arg_fields: ::core::mem::ManuallyDrop::new(#arg_names),)*
                _f: __inner::<#(#generic_idents,)*>,
                _phantom: ::core::marker::PhantomData::<(#(#generic_idents,)*)>,
            })
        }
    };

    TokenStream::from(result)
}
