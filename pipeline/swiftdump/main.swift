import Foundation
// Reads requests.json: [{"l": "de", "k": "card|ord|year|dec|pct|cur|clock", "n": Int, "f": String, "s": String, "h": Int, "m": Int}]
// Writes answers.json: [[String]] in the same order ([] when the request is skipped).
@main struct Dump {
  static func main() {
    let args = CommandLine.arguments
    let inp = args.count > 1 ? args[1] : "requests.json"
    let outp = args.count > 2 ? args[2] : "answers.json"
    guard let d = FileManager.default.contents(atPath: inp),
          let reqs = (try? JSONSerialization.jsonObject(with: d)) as? [[String: Any]] else { print("bad input"); return }
    var out: [[String]] = []
    for r in reqs {
      let l = NumberGrammarLanguage(rawValue: r["l"] as? String ?? "") ?? .german
      let g = NumberGrammars.grammar(for: l)
      let n = r["n"] as? Int ?? 0
      let f = r["f"] as? String ?? ""
      switch r["k"] as? String ?? "" {
      case "card": out.append(g.cardinal(n))
      case "ord": out.append(n >= 1_000_000 || n < 0 ? [] : g.ordinal(n))
      case "year": out.append(g.year(n))
      case "dec": out.append(g.decimal(integer: n, fractionDigits: f))
      case "pct": out.append(g.percent(integer: n, fractionDigits: f))
      case "cur": out.append(g.currency(amount: n, fractionDigits: f, symbol: Character(r["s"] as? String ?? "€")))
      case "clock": out.append(g.clock(hour: r["h"] as? Int ?? 0, minute: r["m"] as? Int ?? 0))
      default: out.append([])
      }
    }
    let data = (try? JSONSerialization.data(withJSONObject: out, options: [])) ?? Data()
    FileManager.default.createFile(atPath: outp, contents: data)
    print("ok", out.count)
  }
}
