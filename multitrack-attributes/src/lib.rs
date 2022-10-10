use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{
    parse_macro_input, spanned::Spanned, Error, FnArg, GenericParam, Ident, ItemFn, Pat,
    TypeParamBound,
};

struct FnInfo<'a> {
    func: &'a ItemFn,
    generic_idents: Vec<&'a Ident>,
    generic_params_no_arch: Vec<&'a GenericParam>,
    generic_idents_no_arch: Vec<&'a Ident>,
    arg_idents: Vec<&'a Ident>,
    arg_types: Vec<Ident>,
    arg_fields: Vec<Ident>,
    arch_ident: &'a Ident,
}

impl<'a> FnInfo<'a> {
    fn from_fn(func: &ItemFn) -> Result<FnInfo, Error> {
        let mut arch_ident = None;

        let mut generic_idents = Vec::new();
        let mut generic_params_no_arch = Vec::new();
        let mut generic_idents_no_arch = Vec::new();
        for param in &func.sig.generics.params {
            match param {
                GenericParam::Type(ty) => {
                    // Look for <_: Arch> generic parameter
                    let mut found_arch = false;
                    if arch_ident.is_none() {
                        for bound in &ty.bounds {
                            if let TypeParamBound::Trait(tr) = bound {
                                if let Some(last) = tr.path.segments.last() {
                                    if last.ident.to_string() == "Arch" {
                                        arch_ident = Some(&ty.ident);
                                        found_arch = true;
                                    }
                                }
                            }
                        }
                    }

                    generic_idents.push(&ty.ident);

                    if !found_arch {
                        generic_params_no_arch.push(param);
                        generic_idents_no_arch.push(&ty.ident);
                    }
                }
                GenericParam::Const(_) => {
                    return Err(Error::new(
                        func.sig.generics.span(),
                        "const generics not supported",
                    ));
                }
                GenericParam::Lifetime(_) => {}
            };
        }

        if arch_ident.is_none() {
            return Err(Error::new(
                func.sig.generics.span(),
                "could not find Arch parameter",
            ));
        }
        let arch_ident = arch_ident.unwrap();

        let mut arg_idents = Vec::with_capacity(func.sig.inputs.len());
        let mut arg_types = Vec::with_capacity(func.sig.inputs.len());
        let mut arg_fields = Vec::with_capacity(func.sig.inputs.len());
        for (i, arg) in func.sig.inputs.iter().enumerate() {
            match arg {
                FnArg::Receiver(_) => {
                    return Err(Error::new(
                        func.span(),
                        "associated functions not supported",
                    ));
                }
                FnArg::Typed(typed) => {
                    if let Pat::Ident(ident) = &*typed.pat {
                        arg_idents.push(&ident.ident);
                        arg_types.push(format_ident!("I{}", i));
                        arg_fields.push(format_ident!("_{}", i));
                    } else {
                        return Err(Error::new(typed.span(), "argument patterns not supported"));
                    }
                }
            }
        }

        Ok(FnInfo {
            func,
            arch_ident,
            generic_idents,
            generic_params_no_arch,
            generic_idents_no_arch,
            arg_idents,
            arg_types,
            arg_fields,
        })
    }

    fn specialize(&self) -> TokenStream2 {
        let attrs = &self.func.attrs;
        let vis = &self.func.vis;
        let sig = &self.func.sig;

        let task = self.task();
        let invoke = self.invoke();

        quote! {
            #(#attrs)*
            #vis #sig {
                #task

                #invoke
            }
        }
    }

    fn task(&self) -> TokenStream2 {
        let generic_idents = &self.generic_idents;
        let generic_params_no_arch = &self.generic_params_no_arch;
        let generic_idents_no_arch = &self.generic_idents_no_arch;
        let arg_types = &self.arg_types;
        let arg_fields = &self.arg_fields;
        let arch_ident = &self.arch_ident;

        let mut inner_sig = self.func.sig.clone();
        inner_sig.ident = format_ident!("__inner");

        let block = &self.func.block;

        quote! {
            struct __Task<#(#generic_idents_no_arch,)* F, #(#arg_types,)*> {
                #(#arg_fields: ::core::mem::ManuallyDrop<#arg_types>,)*
                _f: F,
                _phantom: ::core::marker::PhantomData<(#(#generic_idents_no_arch,)*)>,
            }

            impl<#(#generic_params_no_arch,)* F, #(#arg_types,)* O> ::multitrack::Task for __Task<#(#generic_idents_no_arch,)* F, #(#arg_types),*>
            where
                F: Fn(#(#arg_types),*) -> O,
            {
                type Result = O;

                #[inline(always)]
                fn run<#arch_ident: Arch>(self) -> O {
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
        }
    }

    fn invoke(&self) -> TokenStream2 {
        let generic_idents = &self.generic_idents;
        let generic_idents_no_arch = &self.generic_idents_no_arch;
        let arch_ident = &self.arch_ident;
        let arg_idents = &self.arg_idents;
        let arg_fields = &self.arg_fields;

        quote! {
            #arch_ident::invoke(__Task {
                #(#arg_fields: ::core::mem::ManuallyDrop::new(#arg_idents),)*
                _f: __inner::<#(#generic_idents,)*>,
                _phantom: ::core::marker::PhantomData::<(#(#generic_idents_no_arch,)*)>,
            })
        }
    }
}

#[proc_macro_attribute]
pub fn specialize(_attr: TokenStream, input: TokenStream) -> TokenStream {
    let func = parse_macro_input!(input as ItemFn);

    match FnInfo::from_fn(&func) {
        Ok(info) => info.specialize().into(),
        Err(err) => err.into_compile_error().into(),
    }
}
