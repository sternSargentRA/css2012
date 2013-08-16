function in = inoctave ()
  persistent inout = exist("OCTAVE_VERSION","builtin") != 0;
  in = inout;
endfunction
