//! Byte-level scanner for SMILES / SMARTS strings.
//!
//! SMILES is pure ASCII, so byte indexing is safe and efficient.

use crate::chem::ast::Span;
use crate::error::{SmilesError, SmilesErrorKind};

/// Zero-allocation cursor over a SMILES/SMARTS input string.
pub(crate) struct Scanner<'a> {
    input: &'a str,
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Scanner<'a> {
    /// Create a new scanner over `input`.
    pub fn new(input: &'a str) -> Self {
        Self {
            input,
            bytes: input.as_bytes(),
            pos: 0,
        }
    }

    /// Current byte offset.
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// True when all input has been consumed.
    pub fn is_done(&self) -> bool {
        self.pos >= self.bytes.len()
    }

    /// Look at the current byte as a `char` without consuming it.
    pub fn peek(&self) -> Option<char> {
        self.bytes.get(self.pos).map(|&b| b as char)
    }

    /// Consume the current byte and return it as a `char`.
    pub fn advance(&mut self) -> Option<char> {
        if self.pos < self.bytes.len() {
            let ch = self.bytes[self.pos] as char;
            self.pos += 1;
            Some(ch)
        } else {
            None
        }
    }

    /// Consume the current byte if it matches `expected`, otherwise return an error.
    pub fn expect(&mut self, expected: char) -> Result<(), SmilesError> {
        match self.peek() {
            Some(c) if c == expected => {
                self.pos += 1;
                Ok(())
            }
            Some(c) => Err(self.error(SmilesErrorKind::UnexpectedChar(c))),
            None => Err(self.error(SmilesErrorKind::UnexpectedEnd)),
        }
    }

    /// Consume and return a run of ASCII digits. Returns an empty slice if
    /// the current byte is not a digit.
    pub fn eat_digits(&mut self) -> &'a str {
        let start = self.pos;
        while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_digit() {
            self.pos += 1;
        }
        &self.input[start..self.pos]
    }

    /// Consume a single ASCII digit and return its numeric value.
    pub fn eat_digit(&mut self) -> Option<u8> {
        match self.peek() {
            Some(c) if c.is_ascii_digit() => {
                self.pos += 1;
                Some(c as u8 - b'0')
            }
            _ => None,
        }
    }

    /// Build a [`Span`] from `start` to the current position.
    pub fn span_from(&self, start: usize) -> Span {
        Span::new(start, self.pos)
    }

    /// Build a [`SmilesError`] at the current position.
    pub fn error(&self, kind: SmilesErrorKind) -> SmilesError {
        SmilesError::new(kind, Span::new(self.pos, self.pos + 1), self.input)
    }

    /// Build a [`SmilesError`] with a custom span.
    pub fn error_at(&self, kind: SmilesErrorKind, span: Span) -> SmilesError {
        SmilesError::new(kind, span, self.input)
    }

    /// Peek at the raw byte at the current position (for two-character symbol lookahead).
    pub fn peek_byte(&self) -> Option<&u8> {
        self.bytes.get(self.pos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_peek_advance() {
        let mut s = Scanner::new("ABC");
        assert_eq!(s.peek(), Some('A'));
        assert_eq!(s.advance(), Some('A'));
        assert_eq!(s.peek(), Some('B'));
        assert_eq!(s.advance(), Some('B'));
        assert_eq!(s.advance(), Some('C'));
        assert_eq!(s.advance(), None);
        assert!(s.is_done());
    }

    #[test]
    fn test_empty_input() {
        let s = Scanner::new("");
        assert!(s.is_done());
        assert_eq!(s.peek(), None);
    }

    #[test]
    fn test_expect_ok() {
        let mut s = Scanner::new("C=O");
        assert!(s.expect('C').is_ok());
        assert_eq!(s.pos(), 1);
    }

    #[test]
    fn test_expect_fail() {
        let mut s = Scanner::new("C=O");
        let err = s.expect('N').unwrap_err();
        assert!(matches!(err.kind, SmilesErrorKind::UnexpectedChar('C')));
    }

    #[test]
    fn test_expect_eof() {
        let mut s = Scanner::new("");
        let err = s.expect('C').unwrap_err();
        assert!(matches!(err.kind, SmilesErrorKind::UnexpectedEnd));
    }

    #[test]
    fn test_eat_digits() {
        let mut s = Scanner::new("123abc");
        assert_eq!(s.eat_digits(), "123");
        assert_eq!(s.pos(), 3);
        assert_eq!(s.eat_digits(), "");
        assert_eq!(s.pos(), 3);
    }

    #[test]
    fn test_eat_digit() {
        let mut s = Scanner::new("5X");
        assert_eq!(s.eat_digit(), Some(5));
        assert_eq!(s.eat_digit(), None);
        assert_eq!(s.pos(), 1);
    }

    #[test]
    fn test_span_from() {
        let mut s = Scanner::new("ABCDEF");
        s.advance();
        s.advance();
        let span = s.span_from(0);
        assert_eq!(span, Span::new(0, 2));
    }

    #[test]
    fn test_pos_tracking() {
        let mut s = Scanner::new("C(=O)O");
        for _ in 0..6 {
            s.advance();
        }
        assert_eq!(s.pos(), 6);
        assert!(s.is_done());
    }
}
