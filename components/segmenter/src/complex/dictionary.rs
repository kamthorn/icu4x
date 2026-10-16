// This file is part of ICU4X. For terms of use, please see the file
// called LICENSE at the top level of the ICU4X source tree
// (online at: https://github.com/unicode-org/icu4x/blob/main/LICENSE ).

use crate::grapheme::*;
use crate::indices::*;
use crate::provider::*;
use crate::scaffold::{PotentiallyIllFormedUtf8, RuleBreakType, Utf8, Utf16};
use alloc::vec::Vec;
use icu_collections::char16trie::Char16TrieCursor;
use icu_collections::codepointinvlist::CodePointInversionList;
use zerotrie::cursor::ZeroTrieSimpleAsciiCursor;

/// Thai and Lao vowel signs and marks that cannot start a word.
fn is_dependent(c: u32) -> bool {
    matches!(
        c,
        0x0E30..=0x0E3A | 0x0E45 | 0x0E47..=0x0E4E | 0x0EB0..=0x0EBC | 0x0EC8..=0x0ECE
    )
}

/// Whether a Thai or Lao word can start with `c` (a consonant or a leading vowel). Other
/// characters are not restricted.
fn can_begin_word(c: u32) -> bool {
    match c {
        0x0E00..=0x0E7F => matches!(c, 0x0E01..=0x0E2E | 0x0E40..=0x0E44),
        0x0E80..=0x0EFF => matches!(c, 0x0E81..=0x0EAE | 0x0EC0..=0x0EC4 | 0x0EDC..=0x0EDF),
        _ => true,
    }
}

/// Thai and Lao vowels written before their consonant, which cannot end a word.
fn is_leading_vowel(c: u32) -> bool {
    matches!(c, 0x0E40..=0x0E44 | 0x0EC0..=0x0EC4)
}

enum Cursor<'a> {
    Char16(Char16TrieCursor<'a>),
    Zero {
        alphabet: &'a CodePointInversionList<'a>,
        cursor: ZeroTrieSimpleAsciiCursor<'a>,
    },
}

impl<'a> Cursor<'a> {
    fn new(trie: &'a DictionaryBreakData<'a>) -> Self {
        match trie {
            DictionaryBreakData::Char16Trie(trie) => Cursor::Char16(trie.cursor()),
            DictionaryBreakData::ZeroTrie { alphabet, trie } => Cursor::Zero {
                alphabet,
                cursor: trie.cursor(),
            },
        }
    }

    /// Steps the walk by `c`. Returns whether a word ends here, and whether no longer word can.
    fn step(&mut self, c: u32) -> (bool, bool) {
        match self {
            Cursor::Char16(cursor) => {
                cursor.step32(c);
                (cursor.value().is_some(), cursor.is_empty())
            }
            Cursor::Zero { alphabet, cursor } => {
                cursor.step(alphabet.position32(c).map(|i| i as u8).unwrap_or(0xFF));
                (cursor.take_value().is_some(), cursor.is_empty())
            }
        }
    }
}

/// A token found by [`DictionaryBreakIterator::scan`].
#[derive(Debug)]
struct Token<'data, 's, R: RuleBreakType> {
    /// The end of the token. After the scan, the iterator is at this position.
    end: usize,
    /// Whether the token is a dictionary word.
    word: bool,
    /// Shorter dictionary words at the same position, with the iterator states after them. Only
    /// collected for the Southeast Asian dictionaries.
    shorter: Vec<(usize, R::IterAttr<'s>)>,
    /// A grapheme iterator state at or before the end of the shortest word, if there are shorter
    /// words.
    grapheme_iter: Option<GraphemeClusterBreakIterator<'data, 's, R>>,
}

/// Lifetimes:
/// - `'data` = lifetime of the data
/// - `'s` = lifetime of the string being segmented
#[derive(Debug)]
pub(super) struct DictionaryBreakIterator<'data, 's, R: RuleBreakType> {
    trie: &'data DictionaryBreakData<'data>,
    iter: R::IterAttr<'s>,
    len: usize,
    grapheme_iter: GraphemeClusterBreakIterator<'data, 's, R>,
    /// A token that was scanned to decide the previous one, and is returned next.
    pending: Option<Token<'data, 's, R>>,
    /// Allocations for [`Token::shorter`] to reuse.
    spare: Vec<Vec<(usize, R::IterAttr<'s>)>>,
}

impl<'data, 's, R: RuleBreakType> DictionaryBreakIterator<'data, 's, R> {
    /// How far into the text a chain of up to `words` dictionary words starting at `iter` reaches.
    /// The end of the text, or a character that no word starts with, counts as reaching the end.
    fn reach(&self, iter: &R::IterAttr<'s>, words: u8) -> usize {
        let here = match iter.clone().next() {
            None => return usize::MAX,
            Some((_, c)) if !can_begin_word(c.into()) => return usize::MAX,
            Some((pos, _)) => pos,
        };
        if words == 0 {
            return here;
        }
        let mut iter = iter.clone();
        let mut cursor = Cursor::new(self.trie);
        let mut best = here;
        while let Some((_, c)) = iter.next() {
            let (has_value, is_empty) = cursor.step(c.into());
            if has_value && !matches!(iter.clone().next(), Some((_, c)) if is_dependent(c.into())) {
                best = best.max(self.reach(&iter, words - 1));
                if best == usize::MAX {
                    break;
                }
            }
            if is_empty {
                break;
            }
        }
        best
    }

    fn recycle(&mut self, mut shorter: Vec<(usize, R::IterAttr<'s>)>) {
        if shorter.capacity() > 0 && self.spare.len() < 2 {
            shorter.clear();
            self.spare.push(shorter);
        }
    }

    /// Finds the next token: the longest dictionary word at the current position (also returning
    /// the shorter ones), or the extent of text that no dictionary word matches.
    fn scan(&mut self) -> Option<Token<'data, 's, R>> {
        let mut cursor = Cursor::new(self.trie);
        let keep_shorter = matches!(self.trie, DictionaryBreakData::ZeroTrie { .. });
        let mut not_match = false;
        let mut last_grapheme_offset = 0;
        let mut token_start = None;
        let mut grapheme_iter = None;
        let mut longest_grapheme_iter = None;
        let mut words = self.spare.pop().unwrap_or_default();

        loop {
            let before = self.iter.clone();
            let Some(next) = self.iter.next() else {
                break;
            };
            let start = *token_start.get_or_insert(next.0);
            let (has_value, is_empty) = cursor.step(next.1.into());
            let end = next.0 + R::char_len(next.1);

            if has_value && self.is_word_end(end, &mut last_grapheme_offset) {
                if words.is_empty() {
                    // Grapheme boundaries are only searched forward, so the state after the
                    // shortest word is good for rewinding to any of the words.
                    grapheme_iter = Some(self.grapheme_iter.clone_internal());
                } else {
                    if !keep_shorter {
                        words.clear();
                    }
                    longest_grapheme_iter = Some(self.grapheme_iter.clone_internal());
                }
                words.push((end, self.iter.clone()));
                if is_empty {
                    break;
                }
            } else if is_empty {
                if words.is_empty() {
                    self.recycle(words);
                    // Not found
                    let end = self.end_of_unknown(next, before, start, &mut last_grapheme_offset);
                    return Some(Token {
                        end,
                        word: false,
                        shorter: Vec::new(),
                        grapheme_iter: None,
                    });
                }
                break;
            } else if !has_value {
                // Prefix string is matched
                not_match = true;
            }
        }

        let Some((end, iter)) = words.pop() else {
            self.recycle(words);
            // no match by scanning text
            return not_match.then_some(Token {
                end: self.len,
                word: false,
                shorter: Vec::new(),
                grapheme_iter: None,
            });
        };
        // Rewind to the end of the longest word.
        self.iter = iter;
        let mut kept = None;
        match (grapheme_iter, longest_grapheme_iter) {
            (Some(shortest), Some(longest)) => {
                self.grapheme_iter = longest;
                if !words.is_empty() {
                    kept = Some(shortest);
                }
            }
            (Some(only), None) => self.grapheme_iter = only,
            _ => (),
        }
        Some(Token {
            end,
            word: true,
            shorter: words,
            grapheme_iter: kept,
        })
    }

    /// Moves `last_grapheme_offset` to the first grapheme cluster boundary at or after `end`.
    fn advance_graphemes(&mut self, end: usize, last_grapheme_offset: &mut usize) {
        while *last_grapheme_offset < end {
            match self.grapheme_iter.next() {
                Some(offset) => *last_grapheme_offset = offset,
                None => {
                    *last_grapheme_offset = self.len;
                    break;
                }
            }
        }
    }

    /// Whether a dictionary word can end at `end`: that has to be a grapheme cluster boundary, and
    /// the next character cannot be a vowel sign or mark that belongs to the word.
    fn is_word_end(&mut self, end: usize, last_grapheme_offset: &mut usize) -> bool {
        self.advance_graphemes(end, last_grapheme_offset);
        *last_grapheme_offset == end
            && !matches!(self.iter.clone().next(), Some((_, c)) if is_dependent(c.into()))
    }

    /// Returns the end of a token for which the trie walk found no word, after the character
    /// `failed` made the walk fail. `before` is the iterator state before `failed`, and `start`
    /// the start of the token.
    ///
    /// The end is moved so that it does not split a grapheme cluster, a syllable (vowel signs and
    /// marks stay with their consonant) or a leading vowel from its consonant.
    fn end_of_unknown(
        &mut self,
        failed: (usize, R::CharType),
        before: R::IterAttr<'s>,
        start: usize,
        last_grapheme_offset: &mut usize,
    ) -> usize {
        let leading = is_leading_vowel(failed.1.into());
        if leading && failed.0 > start {
            self.iter = before;
            return failed.0;
        }

        let mut end = failed.0 + R::char_len(failed.1);
        let mut needs_more = leading;
        while let Some((pos, c)) = self.iter.clone().next() {
            if !needs_more && !is_dependent(c.into()) {
                break;
            }
            self.iter.next();
            end = pos + R::char_len(c);
            needs_more = is_leading_vowel(c.into());
        }

        self.advance_graphemes(end, last_grapheme_offset);
        if *last_grapheme_offset > end {
            while let Some((pos, c)) = self.iter.clone().next() {
                if pos >= *last_grapheme_offset {
                    break;
                }
                self.iter.next();
                end = pos + R::char_len(c);
            }
        }
        end
    }
}

/// Implement the [`Iterator`] trait over the segmenter break opportunities of the given string.
/// Please see the [module-level documentation](crate) for its usages.
///
/// [`Iterator`]: core::iter::Iterator
impl<Y: RuleBreakType> Iterator for DictionaryBreakIterator<'_, '_, Y> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let mut token = match self.pending.take() {
            Some(token) => token,
            None => self.scan()?,
        };
        if token.shorter.is_empty() {
            self.recycle(token.shorter);
            return Some(token.end);
        }
        // The longest word is kept if the text after it is a dictionary word, which is then
        // returned by the next call. This is like ICU4C's `ThaiBreakEngine`, which also looks
        // ahead, but it only costs extra work when no word follows.
        let at_boundary = match self.iter.clone().next() {
            None => true,
            Some((_, c)) => !can_begin_word(c.into()),
        };
        let following = if at_boundary { None } else { self.scan() };
        if at_boundary || following.as_ref().is_some_and(|t| t.word) {
            self.pending = following;
            self.recycle(token.shorter);
            return Some(token.end);
        }

        // No dictionary word follows the longest word. Take the shorter word after which the next
        // dictionary word reaches furthest, if any word follows it.
        let mut best = token.end;
        let mut chosen = None;
        for (i, (_, iter)) in token.shorter.iter().enumerate().rev() {
            let reach = self.reach(iter, 1);
            if reach > best {
                best = reach;
                chosen = Some(i);
                if reach == usize::MAX {
                    break;
                }
            }
        }
        let Some(i) = chosen else {
            self.pending = following;
            self.recycle(token.shorter);
            return Some(token.end);
        };
        let (end, iter) = token.shorter.swap_remove(i);
        self.iter = iter;
        if let Some(grapheme_iter) = token.grapheme_iter {
            self.grapheme_iter = grapheme_iter;
        }
        if let Some(following) = following {
            self.recycle(following.shorter);
        }
        self.recycle(token.shorter);
        Some(end)
    }
}

#[derive(Copy, Clone)]
pub(super) struct DictionarySegmenter<'data> {
    dict: &'data DictionaryBreakData<'data>,
    grapheme: GraphemeClusterSegmenterBorrowed<'data>,
}

impl<'data> DictionarySegmenter<'data> {
    pub(super) fn new(
        dict: &'data DictionaryBreakData<'data>,
        grapheme: GraphemeClusterSegmenterBorrowed<'data>,
    ) -> Self {
        // TODO: no way to verify trie data
        Self { dict, grapheme }
    }

    /// Create a dictionary based break iterator for an `str` (a UTF-8 string).
    pub(super) fn segment_str<'s>(
        self,
        input: &'s str,
    ) -> DictionaryBreakIterator<'data, 's, Utf8> {
        let grapheme_iter = self.grapheme.segment_str(input);
        DictionaryBreakIterator {
            trie: self.dict,
            iter: input.char_indices(),
            len: input.len(),
            grapheme_iter,
            pending: None,
            spare: Vec::new(),
        }
    }

    /// Create a dictionary based break iterator for a UTF-8 string.
    pub(super) fn segment_utf8<'s>(
        self,
        input: &'s [u8],
    ) -> DictionaryBreakIterator<'data, 's, PotentiallyIllFormedUtf8> {
        let grapheme_iter = self.grapheme.segment_utf8(input);
        DictionaryBreakIterator {
            trie: self.dict,
            iter: Utf8CharIndices::new(input),
            len: input.len(),
            grapheme_iter,
            pending: None,
            spare: Vec::new(),
        }
    }

    /// Create a dictionary based break iterator for a UTF-16 string.
    pub(super) fn segment_utf16<'s>(
        self,
        input: &'s [u16],
    ) -> DictionaryBreakIterator<'data, 's, Utf16> {
        let grapheme_iter = self.grapheme.segment_utf16(input);
        DictionaryBreakIterator {
            trie: self.dict,
            iter: Utf16Indices::new(input),
            len: input.len(),
            grapheme_iter,
            pending: None,
            spare: Vec::new(),
        }
    }
}

#[cfg(test)]
#[cfg(feature = "serde")]
mod tests {
    use super::*;
    use crate::GraphemeClusterSegmenter;
    use crate::complex::ComplexPayloadsBorrowed;
    use icu_provider::prelude::*;

    use super::super::check_complex;

    #[test]
    fn burmese_dictionary_test() {
        let mut segmenter = ComplexPayloadsBorrowed::new();
        segmenter.with_southeast_asian_dictionaries();
        let segmenter = segmenter.select(ComplexScript::Myanmar).unwrap();

        // From css/css-text/word-break/word-break-normal-my-000.html
        check_complex(
            "မြန်မာစာမြန်မာစာမြန်မာစာ",
            &["မြန်မာ", "စာ", "မြန်မာ", "စာ", "မြန်မာ", "စာ"],
            segmenter,
        );
    }

    #[test]
    fn cj_dictionary_test() {
        let mut segmenter = ComplexPayloadsBorrowed::new();
        segmenter.with_japanese_dictionary();
        let segmenter = segmenter.select(ComplexScript::ChineseOrJapanese).unwrap();

        // Match case
        check_complex("龟山岛龟山岛", &["龟山岛", "龟山岛"], segmenter);

        // Match case, then no match case
        check_complex("エディターエディ", &["エディター", "エディ"], segmenter);
    }

    #[test]
    fn khmer_dictionary_test() {
        let mut segmenter = ComplexPayloadsBorrowed::new();
        segmenter.with_southeast_asian_dictionaries();
        let segmenter = segmenter.select(ComplexScript::Khmer).unwrap();

        check_complex(
            "ភាសាខ្មែរភាសាខ្មែរភាសាខ្មែរ",
            &["ភាសាខ្មែរ", "ភាសាខ្មែរ", "ភាសាខ្មែរ"],
            segmenter,
        );
    }

    #[test]
    fn lao_dictionary_test() {
        let mut segmenter = ComplexPayloadsBorrowed::new();
        segmenter.with_southeast_asian_dictionaries();
        let segmenter = segmenter.select(ComplexScript::Lao).unwrap();
        check_complex(
            "ພາສາລາວພາສາລາວພາສາລາວ",
            &["ພາສາ", "ລາວ", "ພາສາ", "ລາວ", "ພາສາ", "ລາວ"],
            segmenter,
        );
    }

    #[test]
    fn test_dictionary_grapheme_rewind() {
        let response: DataResponse<SegmenterDictionaryAutoV2> = Baked
            .load(DataRequest {
                id: DataIdentifierBorrowed::for_marker_attributes(
                    DataMarkerAttributes::from_str_or_panic("cjdict"),
                ),
                ..Default::default()
            })
            .unwrap();
        let dict_segmenter =
            DictionarySegmenter::new(response.payload.get(), GraphemeClusterSegmenter::new());

        // Test that grapheme_iter is correctly rewound when trie traversal backtracks.
        // In "エディターエディター":
        // 1. "エディター" (15 bytes) is matched as an Intermediate dictionary word at byte 15.
        // 2. Trie traversal continues exploring whether the compound prefix "エディターエディ..."
        //    forms a longer dictionary word, advancing grapheme_iter past byte 15 in the process.
        // 3. When trie traversal eventually hits NoMatch, self.iter rewinds to the last match
        //    at byte 15.
        // 4. Without rewinding grapheme_iter alongside self.iter, on the next call to next()
        //    (starting from byte 15), grapheme_iter is already positioned ahead of byte 15.
        //    When the second word's prefix "エディ" hits an Intermediate match at byte 24,
        //    the grapheme boundary check fails because grapheme_iter skips past 24, causing
        //    incorrect segmentation.
        let s = "エディターエディター";
        let result: Vec<usize> = dict_segmenter.segment_str(s).collect();
        assert_eq!(result, vec![15, 30]);
    }

    #[test]
    fn thai_dictionary_keeps_syllables_together() {
        let mut payloads = ComplexPayloadsBorrowed::new();
        payloads.with_southeast_asian_dictionaries();
        let segmenter = payloads.select(ComplexScript::Thai).unwrap();

        // Words that are not in the dictionary used to be cut between a consonant and its
        // vowel sign or tone mark.
        check_complex(
            "ธนภัทรกับกฤตธีชญ์เล่นเกม",
            &["ธน", "ภัทร", "กับ", "กฤตธี", "ชญ์", "เล่น", "เกม"],
            segmenter,
        );
        // A dictionary word that ends right before a tone mark is not a match.
        check_complex(
            "ตามที่กล่าวมาแล้ว",
            &["ตาม", "ที่", "กล่าว", "มา", "แล้ว"],
            segmenter,
        );

        // When no dictionary word follows the longest match, a shorter match that is followed by
        // one wins (`ดีม` is in the dictionary).
        check_complex(
            "อาหารร้านนี้อร่อยดีมาก",
            &["อาหาร", "ร้าน", "นี้", "อร่อย", "ดี", "มาก"],
            segmenter,
        );
        check_complex("วันนี้อากาศดีมาก", &["วัน", "นี้", "อากาศ", "ดี", "มาก"], segmenter);

        for text in [
            "ลุงไมเคิลเดินไปตลาด",
            "ธนภัทรกับกฤตธีชญ์เล่นเกม",
            "พ่อค้าชื่อวิษณุสุขเกษมขายผลไม้",
            "เธอชอบกินข้าวผัดกะเพราไก่ไข่ดาว",
        ] {
            let mut start = 0;
            for end in segmenter.segment_str(text, GraphemeClusterSegmenter::new(), 0) {
                let segment = &text[start..end];
                let first = segment.chars().next().unwrap() as u32;
                let last = segment.chars().next_back().unwrap() as u32;
                assert!(
                    !is_dependent(first),
                    "{segment} starts with a vowel sign: {text}"
                );
                assert!(
                    !is_leading_vowel(last),
                    "{segment} ends with a leading vowel: {text}"
                );
                start = end;
            }
        }
    }

    #[test]
    fn dictionary_terminal_break() {
        let mut payloads = ComplexPayloadsBorrowed::new();
        payloads.with_southeast_asian_dictionaries();
        payloads.with_japanese_dictionary();

        // Some prefixes end inside a dictionary word; the break at the end of
        // the text must still be emitted.
        for (script, text) in [
            (
                ComplexScript::Thai,
                "ผู้ใช้สามารถติดตั้งโปรแกรมคอมพิวเตอร์ได้ด้วยตนเอง",
            ),
            (ComplexScript::Lao, "ປະເທດລາວມີພົນລະເມືອງປະມານເຈັດລ້ານຄົນ"),
            (ComplexScript::Khmer, "ព្រះរាជាណាចក្រកម្ពុជាជាប្រទេសមួយនៅអាស៊ីអាគ្នេយ៍"),
            (ComplexScript::Myanmar, "မြန်မာနိုင်ငံသည်အရှေ့တောင်အာရှတွင်တည်ရှိသည်"),
            (
                ComplexScript::ChineseOrJapanese,
                "コンピューターソフトウェアエンジニアリング",
            ),
        ] {
            let segmenter = payloads.select(script).unwrap();
            for (i, c) in text.char_indices() {
                let prefix = &text[..i + c.len_utf8()];
                let last = segmenter
                    .segment_str(prefix, GraphemeClusterSegmenter::new(), 0)
                    .last();
                assert_eq!(last, Some(prefix.len()), "{prefix}");
            }
        }
    }
}
