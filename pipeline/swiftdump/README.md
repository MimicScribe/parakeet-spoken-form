Standalone dumper of the app's NumberGrammars (canonical spoken readings for ml_cands.py).

    cp ~/proj/parakeet-transcriber/mimicscribe/Audio/Streaming/NumberGrammar/*.swift src/ && rm src/EnglishNumberGrammar.swift
    cp EnglishStub.swift src/
    swiftc -O -parse-as-library -o dump src/*.swift main.swift
    # then ml_cands.build_table({lang: texts}, "<dir>/dump", "<dir>")

English is stubbed (its grammar depends on DigitRunVerbalizer). Known app bug: SpanishNumberGrammar.ordinal(1_000_000)
traps (index out of range); main.swift skips ordinals >= 1e6.
