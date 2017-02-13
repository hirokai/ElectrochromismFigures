(* ocamlbuild -pkgs ANSITerminal,Batteries verify.byte *)


open Sys
open ANSITerminal
open BatList

type job = {src : string; input : string list; output : string list};;

let check paths =
    let f path =
        let e = file_exists path in
        match e with
            true -> print_string [green] ("Exists: "^path^"\n")
            | false -> print_string [red] ("Not Exists: "^path^"\n") in
    BatList.iter f paths;;

let folder = "/Users/hiroyuki/Documents/Nishizawa Lab/2016 準備中の論文/20160316 Paper Suda electrochromism/20160531-2 Figures managed by git"

let () =
    let j = {src = "plot_debug.py"; input = []; output = ["dist/20170201.pdf";"dist/20170202.pdf"]} in
    if for_all file_exists j.input then begin
        print_string [green] "Prerequisite fulfilled.\n";
        print_string [default] "Running.\n";
        command ("export PYTHONPATH=\""^folder^"\"");
        command ("python "^j.src);
        check j.output
    end else
        print_string [red] "Prerequisite not fulfilled.\n"

