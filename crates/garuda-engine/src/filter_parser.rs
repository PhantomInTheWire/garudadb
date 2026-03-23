use garuda_types::{FilterExpr, LikePattern, ScalarValue, Status, StatusCode, StringMatchExpr};

#[derive(Clone, Debug, PartialEq)]
enum Token {
    Ident(String),
    Number(String),
    Str(String),
    Bool(bool),
    Null,
    Eq,
    Ne,
    Gt,
    Gte,
    Lt,
    Lte,
    Like,
    Contains,
    Is,
    And,
    Or,
    LParen,
    RParen,
}

pub(crate) fn parse_filter(input: &str) -> Result<FilterExpr, Status> {
    let tokens = tokenize(input)?;
    let mut parser = Parser { tokens, pos: 0 };
    let expr = parser.parse_expr()?;

    if parser.pos == parser.tokens.len() {
        return Ok(expr);
    }

    Err(Status::err(
        StatusCode::InvalidArgument,
        "unexpected trailing tokens in filter",
    ))
}

fn tokenize(input: &str) -> Result<Vec<Token>, Status> {
    let chars: Vec<char> = input.chars().collect();
    let mut pos = 0;
    let mut tokens = Vec::new();

    while pos < chars.len() {
        let current = chars[pos];

        if current.is_whitespace() {
            pos += 1;
            continue;
        }

        if let Some(token) = read_symbol_token(&chars, &mut pos)? {
            tokens.push(token);
            continue;
        }

        if current == '\'' || current == '"' {
            tokens.push(read_string_token(&chars, &mut pos)?);
            continue;
        }

        if current.is_ascii_alphabetic() || current == '_' {
            tokens.push(read_identifier_token(&chars, &mut pos));
            continue;
        }

        if current.is_ascii_digit() || current == '-' {
            tokens.push(read_number_token(&chars, &mut pos));
            continue;
        }

        return Err(Status::err(
            StatusCode::InvalidArgument,
            format!("unexpected filter character: {current}"),
        ));
    }

    Ok(tokens)
}

fn read_symbol_token(chars: &[char], pos: &mut usize) -> Result<Option<Token>, Status> {
    match chars[*pos] {
        '(' => {
            *pos += 1;
            Ok(Some(Token::LParen))
        }
        ')' => {
            *pos += 1;
            Ok(Some(Token::RParen))
        }
        '=' => {
            *pos += 1;
            Ok(Some(Token::Eq))
        }
        '!' if chars.get(*pos + 1) == Some(&'=') => {
            *pos += 2;
            Ok(Some(Token::Ne))
        }
        '>' if chars.get(*pos + 1) == Some(&'=') => {
            *pos += 2;
            Ok(Some(Token::Gte))
        }
        '>' => {
            *pos += 1;
            Ok(Some(Token::Gt))
        }
        '<' if chars.get(*pos + 1) == Some(&'=') => {
            *pos += 2;
            Ok(Some(Token::Lte))
        }
        '<' => {
            *pos += 1;
            Ok(Some(Token::Lt))
        }
        '!' => Err(Status::err(
            StatusCode::InvalidArgument,
            "expected '=' after '!'",
        )),
        _ => Ok(None),
    }
}

fn read_string_token(chars: &[char], pos: &mut usize) -> Result<Token, Status> {
    let quote = chars[*pos];
    *pos += 1;
    let start = *pos;

    while *pos < chars.len() && chars[*pos] != quote {
        *pos += 1;
    }

    if *pos >= chars.len() {
        return Err(Status::err(
            StatusCode::InvalidArgument,
            "unterminated string literal",
        ));
    }

    let value: String = chars[start..*pos].iter().collect();
    *pos += 1;
    Ok(Token::Str(value))
}

fn read_identifier_token(chars: &[char], pos: &mut usize) -> Token {
    let start = *pos;
    *pos += 1;

    while *pos < chars.len() && (chars[*pos].is_ascii_alphanumeric() || chars[*pos] == '_') {
        *pos += 1;
    }

    let value: String = chars[start..*pos].iter().collect();
    match value.to_ascii_lowercase().as_str() {
        "and" => Token::And,
        "contains" => Token::Contains,
        "is" => Token::Is,
        "like" => Token::Like,
        "null" => Token::Null,
        "or" => Token::Or,
        "true" => Token::Bool(true),
        "false" => Token::Bool(false),
        _ => Token::Ident(value),
    }
}

fn read_number_token(chars: &[char], pos: &mut usize) -> Token {
    let start = *pos;
    *pos += 1;

    while *pos < chars.len() && (chars[*pos].is_ascii_digit() || chars[*pos] == '.') {
        *pos += 1;
    }

    let value: String = chars[start..*pos].iter().collect();
    Token::Number(value)
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn parse_expr(&mut self) -> Result<FilterExpr, Status> {
        let mut lhs = self.parse_term()?;

        while self.peek() == Some(&Token::Or) {
            self.pos += 1;
            let rhs = self.parse_term()?;
            lhs = FilterExpr::Or(Box::new(lhs), Box::new(rhs));
        }

        Ok(lhs)
    }

    fn parse_term(&mut self) -> Result<FilterExpr, Status> {
        let mut lhs = self.parse_factor()?;

        while self.peek() == Some(&Token::And) {
            self.pos += 1;
            let rhs = self.parse_factor()?;
            lhs = FilterExpr::And(Box::new(lhs), Box::new(rhs));
        }

        Ok(lhs)
    }

    fn parse_factor(&mut self) -> Result<FilterExpr, Status> {
        if self.peek() != Some(&Token::LParen) {
            return self.parse_comparison();
        }

        self.pos += 1;
        let expr = self.parse_expr()?;
        self.expect(Token::RParen)?;
        Ok(expr)
    }

    fn parse_comparison(&mut self) -> Result<FilterExpr, Status> {
        let field = self.parse_field_name()?;
        let operator = self.parse_operator()?;

        match operator {
            Token::Like => {
                let pattern = self.parse_string_literal()?;
                Ok(FilterExpr::StringMatch(
                    field,
                    StringMatchExpr::Like(parse_like_pattern(&pattern)?),
                ))
            }
            Token::Contains => {
                let needle = self.parse_string_literal()?;
                Ok(FilterExpr::StringMatch(
                    field,
                    StringMatchExpr::Contains(needle),
                ))
            }
            Token::Is => {
                self.expect(Token::Null)?;
                Ok(FilterExpr::IsNull(field))
            }
            _ => {
                let value = self.parse_literal()?;
                build_comparison(field, operator, value)
            }
        }
    }

    fn parse_field_name(&mut self) -> Result<String, Status> {
        let Some(Token::Ident(value)) = self.next() else {
            return Err(Status::err(
                StatusCode::InvalidArgument,
                "expected filter field name",
            ));
        };

        Ok(value)
    }

    fn parse_operator(&mut self) -> Result<Token, Status> {
        let Some(operator) = self.next() else {
            return Err(Status::err(
                StatusCode::InvalidArgument,
                "expected filter operator",
            ));
        };

        Ok(operator)
    }

    fn parse_literal(&mut self) -> Result<ScalarValue, Status> {
        let Some(token) = self.next() else {
            return Err(Status::err(
                StatusCode::InvalidArgument,
                "expected filter literal",
            ));
        };

        match token {
            Token::Bool(value) => Ok(ScalarValue::Bool(value)),
            Token::Null => Ok(ScalarValue::Null),
            Token::Str(value) => Ok(ScalarValue::String(value)),
            Token::Number(value) => parse_number_literal(&value),
            _ => Err(Status::err(
                StatusCode::InvalidArgument,
                "expected filter literal",
            )),
        }
    }

    fn parse_string_literal(&mut self) -> Result<String, Status> {
        let Some(Token::Str(value)) = self.next() else {
            return Err(Status::err(
                StatusCode::InvalidArgument,
                "expected string literal",
            ));
        };

        Ok(value)
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn next(&mut self) -> Option<Token> {
        let token = self.tokens.get(self.pos).cloned();

        if token.is_some() {
            self.pos += 1;
        }

        token
    }

    fn expect(&mut self, expected: Token) -> Result<(), Status> {
        let Some(actual) = self.next() else {
            return Err(Status::err(
                StatusCode::InvalidArgument,
                "unexpected end of filter",
            ));
        };

        if actual == expected {
            return Ok(());
        }

        Err(Status::err(
            StatusCode::InvalidArgument,
            "unexpected token in filter",
        ))
    }
}

fn parse_number_literal(value: &str) -> Result<ScalarValue, Status> {
    if value.contains('.') {
        let parsed = value
            .parse::<f64>()
            .map_err(|_| Status::err(StatusCode::InvalidArgument, "invalid float literal"))?;
        return Ok(ScalarValue::Float64(parsed));
    }

    let parsed = value
        .parse::<i64>()
        .map_err(|_| Status::err(StatusCode::InvalidArgument, "invalid integer literal"))?;
    Ok(ScalarValue::Int64(parsed))
}

fn build_comparison(
    field: String,
    operator: Token,
    value: ScalarValue,
) -> Result<FilterExpr, Status> {
    match operator {
        Token::Eq => Ok(FilterExpr::Eq(field, value)),
        Token::Ne => Ok(FilterExpr::Ne(field, value)),
        Token::Gt => Ok(FilterExpr::Gt(field, value)),
        Token::Gte => Ok(FilterExpr::Gte(field, value)),
        Token::Lt => Ok(FilterExpr::Lt(field, value)),
        Token::Lte => Ok(FilterExpr::Lte(field, value)),
        _ => Err(Status::err(
            StatusCode::InvalidArgument,
            "expected comparison operator",
        )),
    }
}

fn parse_like_pattern(pattern: &str) -> Result<LikePattern, Status> {
    let wildcard_count = pattern.chars().filter(|ch| *ch == '%').count();
    if wildcard_count > 1 {
        return Err(Status::err(
            StatusCode::InvalidArgument,
            "LIKE supports at most one '%' wildcard",
        ));
    }

    let Some((prefix, suffix)) = pattern.split_once('%') else {
        return Ok(LikePattern::Exact(pattern.to_string()));
    };

    Ok(LikePattern::PrefixSuffix {
        prefix: prefix.to_string(),
        suffix: suffix.to_string(),
    })
}
