Algorithm options:
-z, --turnpolicy <policy>  - how to resolve ambiguities in path decomposition
-t, --turdsize <n>         - suppress speckles of up to this size (default 2)
-a, --alphamax <n>         - corner threshold parameter (default 1)
-n, --longcurve            - turn off curve optimization
-O, --opttolerance <n>     - curve optimization tolerance (default 0.2)
-u, --unit <n>             - quantize output to 1/unit pixels (default 10)
-d, --debug <n>            - produce debugging output of type n (n=1,2,3)

-z, --turnpolicy <policy>
    specify how to resolve ambiguities in path decomposition. Must be one of
    black, white, right, left, minority, majority, or random. Default is
    minority. Turn policies can be abbreviated by an unambiguous prefix, e.g.,
    one can specify min instead of minority.
-t n, --turdsize n
    suppress speckles of up to this many pixels.

-a n, --alphamax n
    set  the  corner  threshold  parameter.  The  default value is 1. The
    smaller this value, the more sharp corners will be produced. If this
    parameter is 0, then no smoothing will be performed and the output is a
    polygon. If this parameter is greater than 4/3, then  all  corners are
    suppressed and the output is completely smooth.

-n, --longcurve
    turn  off  curve  optimization.  Normally potrace tries to join adjacent
    Bezier curve segments when this is possible. This option disables this
    behavior, resulting in a larger file size.

-O n, --opttolerance n
    set the curve optimization tolerance. The default value is 0.2. Larger
    values allow more consecutive Bezier curve segments  to  be  joined
    together in a single segment, at the expense of accuracy.

-u n, --unit n
    set output quantization. Coordinates in the output are rounded to 1/unit
    pixels. The default of 10 usually gives good results. For some of the debug
    modes, a value of 100 gives more accurate output. This option has no effect
    for the XFig backend,  which  always  rasterizes  to 1/1200  inch,  or for
    the DXF backend. For the GeoJSON backend, this option is only a hint; the
    actual rounding may be more, but not less, accurate than specified.
