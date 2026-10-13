public struct EnglishNumberGrammar: NumberGrammar {
    public init() {}
    public var language: NumberGrammarLanguage { .english }
    public func cardinal(_ n: Int) -> [String] { [] }
    public func decimal(integer: Int, fractionDigits: String) -> [String] { [] }
    public func year(_ n: Int) -> [String] { [] }
    public func clock(hour: Int, minute: Int) -> [String] { [] }
    public func currency(amount: Int, fractionDigits: String, symbol: Character) -> [String] { [] }
    public func percent(integer: Int, fractionDigits: String) -> [String] { [] }
    public func ordinal(_ n: Int) -> [String] { [] }
    public func parse(_ words: [String]) -> NumericValue? { nil }
    public var closedVocabularySample: [String] { [] }
}
