use multitrack::{Arch, select, generic};

fn main() {

}

#[select]
fn _f<A: Arch>() {
    _g::<A>();
}

#[generic]
fn _g<A: Arch>() {

}
