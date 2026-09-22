// Safe expression language shared with the Python engine (SPEC.md §3).
// Hand-written recursive-descent parser; never uses eval or Function().
// Mirrors python/dsvm/expr.py token for token; keep the two in step.

export class ExprError extends Error {
  constructor(msg) { super(msg); this.name = "ExprError"; }
}

function roundHalfAway(a) {
  return a >= 0 ? Math.floor(a + 0.5) : -Math.floor(-a + 0.5);
}

function checkFinite(name, v) {
  if (!Number.isFinite(v)) throw new ExprError(`${name} produced a non-finite value (${v})`);
  return v;
}

function fnPow(a, b) {
  const v = Math.pow(a, b);
  if (Number.isNaN(v)) throw new ExprError(`pow(${a}, ${b}) is not a real number`);
  return checkFinite("pow", v);
}

export const FUNCTIONS = {
  min: (...a) => Math.min(...a),
  max: (...a) => Math.max(...a),
  abs: Math.abs,
  floor: Math.floor,
  ceil: Math.ceil,
  round: roundHalfAway,
  sqrt: (a) => { if (a < 0) throw new ExprError(`sqrt of negative number (${a})`); return Math.sqrt(a); },
  exp: (a) => checkFinite("exp", Math.exp(a)),
  log: (a) => { if (a <= 0) throw new ExprError(`log of non-positive number (${a})`); return Math.log(a); },
  pow: fnPow,
  if: (c, a, b) => (c !== 0 ? a : b),
};
export const FUNCTION_ARITY = { min: null, max: null, abs: 1, floor: 1, ceil: 1, round: 1, sqrt: 1, exp: 1, log: 1, pow: 2, if: 3 };
export const RESERVED = new Set([...Object.keys(FUNCTIONS), "t", "T", "and", "or", "not"]);

const TOKEN_RE = /(?<num>\d+\.\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?|\d+(?:[eE][+-]?\d+)?)|(?<id>[A-Za-z_][A-Za-z0-9_]*)|(?<op><=|>=|==|!=|&&|\|\||[-+*/%^()<>?:,!])/y;

export function tokenize(src) {
  const toks = [];
  let pos = 0;
  while (pos < src.length) {
    if (/\s/.test(src[pos])) { pos += 1; continue; }
    TOKEN_RE.lastIndex = pos;
    const m = TOKEN_RE.exec(src);
    if (!m) throw new ExprError(`unexpected character '${src[pos]}' at position ${pos} in '${src}'`);
    const kind = m.groups.num !== undefined ? "num" : m.groups.id !== undefined ? "id" : "op";
    toks.push({ kind, text: m[0], pos });
    pos = TOKEN_RE.lastIndex;
  }
  toks.push({ kind: "end", text: "", pos: src.length });
  return toks;
}

class Parser {
  constructor(src, variables) {
    this.src = src;
    this.toks = tokenize(src);
    this.i = 0;
    this.variables = new Set(variables);
    this.used = new Set();
  }
  peek() { return this.toks[this.i]; }
  take(text) {
    const tok = this.toks[this.i];
    if (text !== undefined && tok.text !== text) {
      throw new ExprError(`expected '${text}' at position ${tok.pos} in '${this.src}', found '${tok.text}'`);
    }
    this.i += 1;
    return tok;
  }
  accept(...texts) {
    const tok = this.toks[this.i];
    if ((tok.kind === "op" || tok.kind === "id") && texts.includes(tok.text)) { this.i += 1; return tok; }
    return null;
  }
  parse() {
    const node = this.ternary();
    if (this.peek().kind !== "end") {
      const tok = this.peek();
      throw new ExprError(`unexpected '${tok.text}' at position ${tok.pos} in '${this.src}'`);
    }
    return node;
  }
  ternary() {
    const cond = this.or_();
    if (this.accept("?")) {
      const a = this.ternary();
      this.take(":");
      const b = this.ternary();
      return (env) => (cond(env) !== 0 ? a(env) : b(env));
    }
    return cond;
  }
  or_() {
    let left = this.and_();
    while (this.accept("or", "||")) {
      const right = this.and_();
      const l = left;
      left = (env) => (l(env) !== 0 || right(env) !== 0 ? 1 : 0);
    }
    return left;
  }
  and_() {
    let left = this.not_();
    while (this.accept("and", "&&")) {
      const right = this.not_();
      const l = left;
      left = (env) => (l(env) !== 0 && right(env) !== 0 ? 1 : 0);
    }
    return left;
  }
  not_() {
    if (this.accept("not", "!")) {
      const inner = this.not_();
      return (env) => (inner(env) !== 0 ? 0 : 1);
    }
    return this.compare();
  }
  compare() {
    const left = this.sum();
    const tok = this.accept("<", "<=", ">", ">=", "==", "!=");
    if (!tok) return left;
    const right = this.sum();
    const cmp = {
      "<": (a, b) => a < b, "<=": (a, b) => a <= b, ">": (a, b) => a > b,
      ">=": (a, b) => a >= b, "==": (a, b) => a === b, "!=": (a, b) => a !== b,
    }[tok.text];
    return (env) => (cmp(left(env), right(env)) ? 1 : 0);
  }
  sum() {
    let left = this.product();
    for (;;) {
      const tok = this.accept("+", "-");
      if (!tok) return left;
      const right = this.product();
      const l = left;
      left = tok.text === "+" ? (env) => l(env) + right(env) : (env) => l(env) - right(env);
    }
  }
  product() {
    let left = this.unary();
    for (;;) {
      const tok = this.accept("*", "/", "%");
      if (!tok) return left;
      const right = this.unary();
      const l = left;
      if (tok.text === "*") left = (env) => l(env) * right(env);
      else if (tok.text === "/") {
        left = (env) => { const d = right(env); if (d === 0) throw new ExprError("division by zero"); return l(env) / d; };
      } else {
        left = (env) => { const d = right(env); if (d === 0) throw new ExprError("modulo by zero"); return l(env) % d; };
      }
    }
  }
  unary() {
    if (this.accept("-")) {
      const inner = this.unary();
      return (env) => -inner(env);
    }
    return this.power();
  }
  power() {
    const base = this.atom();
    if (this.accept("^")) {
      const exp = this.unary();
      return (env) => fnPow(base(env), exp(env));
    }
    return base;
  }
  atom() {
    const tok = this.peek();
    if (tok.kind === "num") {
      this.i += 1;
      const v = Number(tok.text);
      return () => v;
    }
    if (tok.kind === "id") {
      this.i += 1;
      const name = tok.text;
      if (this.accept("(")) {
        if (!(name in FUNCTIONS)) throw new ExprError(`unknown function '${name}' at position ${tok.pos} in '${this.src}'`);
        const args = [];
        if (!this.accept(")")) {
          args.push(this.ternary());
          while (this.accept(",")) args.push(this.ternary());
          this.take(")");
        }
        const arity = FUNCTION_ARITY[name];
        if (arity === null) {
          if (args.length < 1) throw new ExprError(`${name}() needs at least one argument in '${this.src}'`);
        } else if (args.length !== arity) {
          throw new ExprError(`${name}() takes ${arity} argument(s), got ${args.length} in '${this.src}'`);
        }
        const fn = FUNCTIONS[name];
        return (env) => fn(...args.map((a) => a(env)));
      }
      if (name === "and" || name === "or" || name === "not") {
        throw new ExprError(`unexpected keyword '${name}' at position ${tok.pos} in '${this.src}'`);
      }
      if (!this.variables.has(name) && name !== "t" && name !== "T") {
        throw new ExprError(`unknown variable '${name}' in '${this.src}'; known: t, T, ${[...this.variables].sort().join(", ")}`);
      }
      this.used.add(name);
      return (env) => env[name];
    }
    if (tok.kind === "op" && tok.text === "(") {
      this.i += 1;
      const inner = this.ternary();
      this.take(")");
      return inner;
    }
    if (tok.kind === "end") throw new ExprError(`unexpected end of expression in '${this.src}'`);
    throw new ExprError(`unexpected '${tok.text}' at position ${tok.pos} in '${this.src}'`);
  }
}

export class Expression {
  /** new Expression("0.4*x", ["x"]).evaluate({x: 5, t: 1, T: 10}) */
  constructor(source, variables) {
    if (typeof source === "boolean") throw new ExprError("boolean is not a valid expression");
    if (typeof source === "number") source = String(source);
    if (typeof source !== "string") throw new ExprError(`expression must be a string or number, got ${typeof source}`);
    if (source.trim() === "") throw new ExprError("empty expression");
    this.source = source;
    const parser = new Parser(source, variables);
    this.node = parser.parse();
    /** identifiers the expression reads (state variable names, 't', 'T') */
    this.dependsOn = new Set(parser.used);
  }
  evaluate(env) {
    const v = this.node(env);
    return checkFinite(this.source, Number(v));
  }
}

export function isValidIdentifier(name) {
  return /^[A-Za-z_][A-Za-z0-9_]*$/.test(name) && !RESERVED.has(name);
}
