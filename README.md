# moska
A repo for simulating moska games, and trying different play algorithms.

This is designed for simulations of moska games with different algorithms, and playing as a human can be quite tedious.

## TODO LIST
- Hide `Turns` class from `/Player`, so that a play request with arguments is made to `Game` and the request is processed and the proper Turn called.
- Move re-usable Player functions away from BasePlayer or its subclasses to make reusing easier.
- Create TESTS!!
- Manage the different Game processes with `multiprocessing.Pool`
- Create a benchmarking system, that records each games initial order, final ranking. Then create an analyzing system.
- If benchmarking works, make it possible to run games without logging to increase speed.
- thread : player -mapping could probably be made global, and used to prevent accidental modification of moskaGame and other attributes