import logging
import re
import warnings
from pathlib import Path
from typing import Iterable, List, Optional, Union

import g2p_en
import jamo
from packaging.version import parse as V
from typeguard import check_argument_types

from espnet2.text.abs_tokenizer import AbsTokenizer

g2p_choices = [
    None,
    "g2p_en",
    "g2p_en_no_space",
    "pyopenjtalk",
    "pyopenjtalk_kana",
    "pyopenjtalk_accent",
    "pyopenjtalk_accent_with_pause",
    "pyopenjtalk_prosody",
    "pyopenjtalk_prosody_sp_token",
    "pypinyin_g2p",
    "pypinyin_g2p_phone",
    "pypinyin_g2p_phone_without_prosody",
    "espeak_ng_arabic",
    "espeak_ng_german",
    "espeak_ng_french",
    "espeak_ng_spanish",
    "espeak_ng_russian",
    "espeak_ng_greek",
    "espeak_ng_finnish",
    "espeak_ng_hungarian",
    "espeak_ng_dutch",
    "espeak_ng_english_us_vits",
    "espeak_ng_hindi",
    "espeak_ng_italian",
    "espeak_ng_ukrainian",
    "espeak_ng_polish",
    "g2pk",
    "g2pk_no_space",
    "g2pk_explicit_space",
    "korean_jaso",
    "korean_jaso_no_space",
    "g2p_is",
]


def split_by_space(text) -> List[str]:
    if "   " in text:
        text = text.replace("   ", " <space> ")
        return [c.replace("<space>", " ") for c in text.split(" ")]
    else:
        return text.split(" ")


def pyopenjtalk_g2p(text) -> List[str]:
    import pyopenjtalk

    # phones is a str object separated by space
    phones = pyopenjtalk.g2p(text, kana=False)
    phones = phones.split(" ")
    return phones


def _extract_fullcontext_label(text):
    import pyopenjtalk

    if V(pyopenjtalk.__version__) >= V("0.3.0"):
        return pyopenjtalk.make_label(pyopenjtalk.run_frontend(text))
    else:
        return pyopenjtalk.run_frontend(text)[1]


def pyopenjtalk_g2p_accent(text) -> List[str]:
    phones = []
    for labels in _extract_fullcontext_label(text):
        p = re.findall(r"\-(.*?)\+.*?\/A:([0-9\-]+).*?\/F:.*?_([0-9]+)", labels)
        if len(p) == 1:
            phones += [p[0][0], p[0][2], p[0][1]]
    return phones


def pyopenjtalk_g2p_accent_with_pause(text) -> List[str]:
    phones = []
    for labels in _extract_fullcontext_label(text):
        if labels.split("-")[1].split("+")[0] == "pau":
            phones += ["pau"]
            continue
        p = re.findall(r"\-(.*?)\+.*?\/A:([0-9\-]+).*?\/F:.*?_([0-9]+)", labels)
        if len(p) == 1:
            phones += [p[0][0], p[0][2], p[0][1]]
    return phones


def pyopenjtalk_g2p_kana(text) -> List[str]:
    import pyopenjtalk

    kanas = pyopenjtalk.g2p(text, kana=True)
    return list(kanas)


def pyopenjtalk_g2p_prosody(text: str, drop_unvoiced_vowels: bool = True) -> List[str]:
    """Extract phoneme + prosoody symbol sequence from input full-context labels.

    The algorithm is based on `Prosodic features control by symbols as input of
    sequence-to-sequence acoustic modeling for neural TTS`_ with some r9y9's tweaks.

    Args:
        text (str): Input text.
        drop_unvoiced_vowels (bool): whether to drop unvoiced vowels.

    Returns:
        List[str]: List of phoneme + prosody symbols.

    Examples:
        >>> from espnet2.text.phoneme_tokenizer import pyopenjtalk_g2p_prosody
        >>> pyopenjtalk_g2p_prosody("こんにちは。")
        ['^', 'k', 'o', '[', 'N', 'n', 'i', 'ch', 'i', 'w', 'a', '$']

    .. _`Prosodic features control by symbols as input of sequence-to-sequence acoustic
        modeling for neural TTS`: https://doi.org/10.1587/transinf.2020EDP7104

    """
    labels = _extract_fullcontext_label(text)
    N = len(labels)

    phones = []
    for n in range(N):
        lab_curr = labels[n]

        # current phoneme
        p3 = re.search(r"\-(.*?)\+", lab_curr).group(1)

        # deal unvoiced vowels as normal vowels
        if drop_unvoiced_vowels and p3 in "AEIOU":
            p3 = p3.lower()

        # deal with sil at the beginning and the end of text
        if p3 == "sil":
            assert n == 0 or n == N - 1
            if n == 0:
                phones.append("^")
            elif n == N - 1:
                # check question form or not
                e3 = _numeric_feature_by_regex(r"!(\d+)_", lab_curr)
                if e3 == 0:
                    phones.append("$")
                elif e3 == 1:
                    phones.append("?")
            continue
        elif p3 == "pau":
            phones.append("_")
            continue
        else:
            phones.append(p3)

        # accent type and position info (forward or backward)
        a1 = _numeric_feature_by_regex(r"/A:([0-9\-]+)\+", lab_curr)
        a2 = _numeric_feature_by_regex(r"\+(\d+)\+", lab_curr)
        a3 = _numeric_feature_by_regex(r"\+(\d+)/", lab_curr)

        # number of mora in accent phrase
        f1 = _numeric_feature_by_regex(r"/F:(\d+)_", lab_curr)

        a2_next = _numeric_feature_by_regex(r"\+(\d+)\+", labels[n + 1])
        # accent phrase border
        if a3 == 1 and a2_next == 1 and p3 in "aeiouAEIOUNcl":
            phones.append("#")
        # pitch falling
        elif a1 == 0 and a2_next == a2 + 1 and a2 != f1:
            phones.append("]")
        # pitch rising
        elif a2 == 1 and a2_next == 2:
            phones.append("[")

    return phones


def pyopenjtalk_g2p_prosody_sp_token(text: str, drop_unvoiced_vowels: bool = True) -> List[str]:
    """
    Process text with special tags (e.g., <tag>...</tag>) by splitting, extracting prosody symbols,
    and combining results with markers for boundary positions, matching full input sequence.

    Args:
        text (str): Input text with special markers (e.g., <tag>, </tag>).
        drop_unvoiced_vowels (bool): Whether to drop unvoiced vowels in the phoneme sequence.

    Returns:
        Tuple[List[str], List[str]]: Combined phoneme + prosody symbol sequence with boundary markers,
        and a list of special tokens.

    Example:
        >>> pyopenjtalk_g2p_prosody_sp_token("時刻はまもなく<tag>2時</tag>5分です")
        (['^', 'j', 'i', ']', 'k', 'o', 'k', 'u', 'w', 'a', '#', 'm', 'a', '[', 'm', 'o', ']', 'n', 'a', 'k', 'u', '#', '<tag>', 'n', 'i', ']', 'j', 'i', '</tag>', '#', 'g', 'o', ']', 'f', 'u', 'N', 'd', 'e', 's', 'u', '$'], ['<tag>', '</tag>'])
    """
    special_symbols = re.findall(r'<.*?>', text)
    
    clean_text = re.sub(r'<[^>]+>', '', text)
    clean_phones = pyopenjtalk_g2p_prosody(clean_text, drop_unvoiced_vowels)
    
    emp_phones = pyopenjtalk_g2p_prosody(text, drop_unvoiced_vowels)
    emp_phones = _replace_continuous_pattern(emp_phones, special_symbols)
    emp_phones = _dp_with_shortest_alignment(emp_phones, clean_phones, special_symbols)
    return emp_phones


def extract_special_symbols(input_string: str) -> List[str]:
    """
    Extract the contents within special symbols '<' and '>' in a given string.

    Args:
        input_string (str): The input string containing special symbols.

    Returns:
        List[str]: A list of extracted contents within '<' and '>'.
    """
    return re.findall(r'<(.*?)>', input_string)


def _replace_continuous_pattern(input_list: List[str], special_symbols: List[str]) -> List[str]:
    """
    Replace continuous patterns ['_', 'i', ']', 'i', '_'] or ['_', 'i', ']', 'i', '_', 'i', ']', 'i', '_']
    with symbols from the provided `special_symbols` list in sequence.

    Args:
        input_list (List[str]): The input list of phonemes and symbols.
        special_symbols (List[str]): A list of special symbols to insert in sequence.

    Returns:
        List[str]: The modified list with replacements applied.
    """
    short_pattern = ['_', 'i', ']', 'i', '_']
    long_pattern = ['_', 'i', ']', 'i', '_', 'i', ']', 'i', '_']
    output_list = []
    i = 0
    symbol_index = 0  # Index to track the current symbol in special_symbols

    while i < len(input_list):
        if input_list[i:i+len(long_pattern)] == long_pattern:
            # Insert two symbols for the long pattern
            output_list.append(special_symbols[symbol_index % len(special_symbols)])
            symbol_index += 1
            output_list.append(special_symbols[symbol_index % len(special_symbols)])
            symbol_index += 1
            i += len(long_pattern)
        elif input_list[i:i+len(short_pattern)] == short_pattern:
            # Insert one symbol for the short pattern
            output_list.append(special_symbols[symbol_index % len(special_symbols)])
            symbol_index += 1
            i += len(short_pattern)
        else:
            output_list.append(input_list[i])
            i += 1

    return output_list


def _preprocess_continuous_symbols(a, special_symbols):
    """
    將連續的特殊符號替換為唯一的單一標記，並記錄原始連續符號的位置與內容。
    
    :param a: 序列 a。
    :param special_symbols: 特殊符號集合。
    :return: 處理後的序列和連續特殊符號的替換記錄。
    """
    processed_a = []
    replacements = {}
    i = 0
    marker_id = 0  # 用於生成唯一標記

    while i < len(a):
        if a[i] in special_symbols:
            start = i
            while i < len(a) and a[i] in special_symbols:
                i += 1
            # 替換連續特殊符號為唯一標記
            replacement_marker = f"<SYM_{marker_id}>"
            marker_id += 1
            processed_a.append(replacement_marker)
            replacements[replacement_marker] = a[start:i]  # 記錄替換
        else:
            processed_a.append(a[i])
            i += 1
    return processed_a, replacements


def _replace_back(result, replacements):
    """
    將替換符號還原為原始連續符號。
    
    :param result: 包含替換符號的結果序列。
    :param replacements: 替換記錄。
    :return: 還原後的序列。
    """
    restored_result = []
    for item in result:
        if item in replacements:
            restored_result.extend(replacements[item])  # 替換回原始連續符號
        else:
            restored_result.append(item)
    return restored_result


def _dp_with_shortest_alignment(a, b, special_symbols):
    """
    基於 DP 的多種特殊符號插入算法，處理連續特殊符號，並插入到最短匹配的位置。

    :param a: 序列 a。
    :param b: 序列 b。
    :param special_symbols: 特殊符號集合。
    :return: 包含特殊符號的結果序列。
    """
    # 預處理連續特殊符號
    processed_a, replacements = _preprocess_continuous_symbols(a, special_symbols)

    m, n = len(processed_a), len(b)
    dp = [[float('inf')] * (n + 1) for _ in range(m + 1)]

    # 初始化 DP 表
    dp[0][0] = 0
    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] + 1  # 插入所有特殊符號
    for j in range(1, n + 1):
        dp[0][j] = j  # 保留所有 b 的元素

    # 填充 DP 表
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if processed_a[i - 1] == b[j - 1]:  # 元素匹配
                dp[i][j] = dp[i - 1][j - 1]
            elif processed_a[i - 1] in replacements:  # 插入特殊符號
                dp[i][j] = min(dp[i][j], dp[i - 1][j] + 1)
            else:  # 跳過 processed_a[i-1] 或保留 b[j-1]
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1] + 1)

    # 回溯構造結果
    i, j = m, n
    result = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and processed_a[i - 1] == b[j - 1]:  # 匹配
            result.append(b[j - 1])
            i -= 1
            j -= 1
        elif i > 0 and processed_a[i - 1] in replacements:  # 插入特殊符號
            result.append(processed_a[i - 1])
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:  # 保留 b[j-1]
            result.append(b[j - 1])
            j -= 1
        else:  # 跳過 processed_a[i-1]
            i -= 1

    # 還原連續符號
    return _replace_back(result[::-1], replacements)


def _numeric_feature_by_regex(regex, s):
    match = re.search(regex, s)
    if match is None:
        return -50
    return int(match.group(1))


# if __name__ == "__main__":
#     # 使用例
#     text = "17日の<e>マーリンズ戦で</e><e>今シーズン</e>48号となるホームランを打ち"
#     emp_phn = pyopenjtalk_g2p_prosody_sp_token(text)
#     print(emp_phn)


def pypinyin_g2p(text) -> List[str]:
    from pypinyin import Style, pinyin

    phones = [phone[0] for phone in pinyin(text, style=Style.TONE3)]
    return phones


def pypinyin_g2p_phone(text) -> List[str]:
    from pypinyin import Style, pinyin
    from pypinyin.style._utils import get_finals, get_initials

    phones = [
        p
        for phone in pinyin(text, style=Style.TONE3)
        for p in [
            get_initials(phone[0], strict=True),
            get_finals(phone[0][:-1], strict=True) + phone[0][-1]
            if phone[0][-1].isdigit()
            else get_finals(phone[0], strict=True)
            if phone[0][-1].isalnum()
            else phone[0],
        ]
        # Remove the case of individual tones as a phoneme
        if len(p) != 0 and not p.isdigit()
    ]
    return phones


def pypinyin_g2p_phone_without_prosody(text) -> List[str]:
    from pypinyin import Style, pinyin
    from pypinyin.style._utils import get_finals, get_initials

    phones = []
    for phone in pinyin(text, style=Style.NORMAL, strict=False):
        initial = get_initials(phone[0], strict=False)
        final = get_finals(phone[0], strict=False)
        if len(initial) != 0:
            if initial in ["x", "y", "j", "q"]:
                if final == "un":
                    final = "vn"
                elif final == "uan":
                    final = "van"
                elif final == "u":
                    final = "v"
            if final == "ue":
                final = "ve"
            phones.append(initial + "_" + final)
        else:
            phones.append(final)
    return phones


class G2p_en:
    """On behalf of g2p_en.G2p.

    g2p_en.G2p isn't pickalable and it can't be copied to the other processes
    via multiprocessing module.
    As a workaround, g2p_en.G2p is instantiated upon calling this class.

    """

    def __init__(self, no_space: bool = False):
        self.no_space = no_space
        self.g2p = None

    def __call__(self, text) -> List[str]:
        if self.g2p is None:
            self.g2p = g2p_en.G2p()

        phones = self.g2p(text)
        if self.no_space:
            # remove space which represents word serapater
            phones = list(filter(lambda s: s != " ", phones))
        return phones


class G2pk:
    """On behalf of g2pk.G2p.

    g2pk.G2p isn't pickalable and it can't be copied to the other processes
    via multiprocessing module.
    As a workaround, g2pk.G2p is instantiated upon calling this class.

    """

    def __init__(
        self,
        descritive=False,
        group_vowels=False,
        to_syl=False,
        no_space=False,
        explicit_space=False,
        space_symbol="<space>",
    ):
        self.descritive = descritive
        self.group_vowels = group_vowels
        self.to_syl = to_syl
        self.no_space = no_space
        self.explicit_space = explicit_space
        self.space_symbol = space_symbol
        self.g2p = None

    def __call__(self, text) -> List[str]:
        if self.g2p is None:
            import g2pk

            self.g2p = g2pk.G2p()

        phones = list(
            self.g2p(
                text,
                descriptive=self.descritive,
                group_vowels=self.group_vowels,
                to_syl=self.to_syl,
            )
        )
        if self.no_space:
            # remove space which represents word serapater
            phones = list(filter(lambda s: s != " ", phones))

        if self.explicit_space:
            # replace space as explicit space symbol
            phones = list(map(lambda s: s if s != " " else self.space_symbol, phones))
        return phones


class Jaso:
    PUNC = "!'(),-.:;?"
    SPACE = " "

    JAMO_LEADS = "".join([chr(_) for _ in range(0x1100, 0x1113)])
    JAMO_VOWELS = "".join([chr(_) for _ in range(0x1161, 0x1176)])
    JAMO_TAILS = "".join([chr(_) for _ in range(0x11A8, 0x11C3)])

    VALID_CHARS = JAMO_LEADS + JAMO_VOWELS + JAMO_TAILS + PUNC + SPACE

    def __init__(self, space_symbol=" ", no_space=False):
        self.space_symbol = space_symbol
        self.no_space = no_space

    def _text_to_jaso(self, line: str) -> List[str]:
        jasos = list(jamo.hangul_to_jamo(line))
        return jasos

    def _remove_non_korean_characters(self, tokens):
        new_tokens = [token for token in tokens if token in self.VALID_CHARS]
        return new_tokens

    def __call__(self, text) -> List[str]:
        graphemes = [x for x in self._text_to_jaso(text)]
        graphemes = self._remove_non_korean_characters(graphemes)

        if self.no_space:
            graphemes = list(filter(lambda s: s != " ", graphemes))
        else:
            graphemes = [x if x != " " else self.space_symbol for x in graphemes]
        return graphemes


class Phonemizer:
    """Phonemizer module for various languages.

    This is wrapper module of https://github.com/bootphon/phonemizer.
    You can define various g2p modules by specifying options for phonemizer.

    See available options:
        https://github.com/bootphon/phonemizer/blob/master/phonemizer/phonemize.py#L32

    """

    def __init__(
        self,
        backend,
        word_separator: Optional[str] = None,
        syllable_separator: Optional[str] = None,
        phone_separator: Optional[str] = " ",
        strip=False,
        split_by_single_token: bool = False,
        **phonemizer_kwargs,
    ):
        # delayed import
        from phonemizer.backend import BACKENDS
        from phonemizer.separator import Separator

        self.separator = Separator(
            word=word_separator,
            syllable=syllable_separator,
            phone=phone_separator,
        )

        # define logger to suppress the warning in phonemizer
        logger = logging.getLogger("phonemizer")
        logger.setLevel(logging.ERROR)
        self.phonemizer = BACKENDS[backend](
            **phonemizer_kwargs,
            logger=logger,
        )
        self.strip = strip
        self.split_by_single_token = split_by_single_token

    def __call__(self, text) -> List[str]:
        tokens = self.phonemizer.phonemize(
            [text],
            separator=self.separator,
            strip=self.strip,
            njobs=1,
        )[0]
        if not self.split_by_single_token:
            return tokens.split()
        else:
            # "a: ab" -> ["a", ":", "<space>",  "a", "b"]
            # TODO(kan-bayashi): space replacement should be dealt in PhonemeTokenizer
            return [c.replace(" ", "<space>") for c in tokens]


class IsG2p:  # pylint: disable=too-few-public-methods
    """Minimal wrapper for https://github.com/grammatek/ice-g2p

    The g2p module uses a Bi-LSTM model along with
    a pronunciation dictionary to generate phonemization
    Unfortunately does not support multi-thread phonemization as of yet
    """

    def __init__(
        self,
        dialect: str = "standard",
        syllabify: bool = True,
        word_sep: str = ",",
        use_dict: bool = True,
    ):
        self.dialect = dialect
        self.syllabify = syllabify
        self.use_dict = use_dict
        from ice_g2p.transcriber import Transcriber

        self.transcriber = Transcriber(
            use_dict=self.use_dict,
            syllab_symbol=".",
            stress_label=True,
            word_sep=word_sep,
            lang_detect=True,
        )

    def __call__(self, text) -> List[str]:
        return self.transcriber.transcribe(text).split()


class PhonemeTokenizer(AbsTokenizer):
    def __init__(
        self,
        g2p_type: Union[None, str],
        non_linguistic_symbols: Union[Path, str, Iterable[str]] = None,
        space_symbol: str = "<space>",
        remove_non_linguistic_symbols: bool = False,
    ):
        assert check_argument_types()
        if g2p_type is None:
            self.g2p = split_by_space
        elif g2p_type == "g2p_en":
            self.g2p = G2p_en(no_space=False)
        elif g2p_type == "g2p_en_no_space":
            self.g2p = G2p_en(no_space=True)
        elif g2p_type == "pyopenjtalk":
            self.g2p = pyopenjtalk_g2p
        elif g2p_type == "pyopenjtalk_kana":
            self.g2p = pyopenjtalk_g2p_kana
        elif g2p_type == "pyopenjtalk_accent":
            self.g2p = pyopenjtalk_g2p_accent
        elif g2p_type == "pyopenjtalk_accent_with_pause":
            self.g2p = pyopenjtalk_g2p_accent_with_pause
        elif g2p_type == "pyopenjtalk_prosody":
            self.g2p = pyopenjtalk_g2p_prosody
        elif g2p_type == "pyopenjtalk_prosody_sp_token":
            self.g2p = pyopenjtalk_g2p_prosody_sp_token
        elif g2p_type == "pypinyin_g2p":
            self.g2p = pypinyin_g2p
        elif g2p_type == "pypinyin_g2p_phone":
            self.g2p = pypinyin_g2p_phone
        elif g2p_type == "pypinyin_g2p_phone_without_prosody":
            self.g2p = pypinyin_g2p_phone_without_prosody
        elif g2p_type == "espeak_ng_arabic":
            self.g2p = Phonemizer(
                language="ar",
                backend="espeak",
                with_stress=True,
                preserve_punctuation=True,
            )
        elif g2p_type == "espeak_ng_german":
            self.g2p = Phonemizer(
                language="de",
                backend="espeak",
                with_stress=True,
                preserve_punctuation=True,
            )
        elif g2p_type == "espeak_ng_french":
            self.g2p = Phonemizer(
                language="fr-fr",
                backend="espeak",
                with_stress=True,
                preserve_punctuation=True,
            )
        elif g2p_type == "espeak_ng_spanish":
            self.g2p = Phonemizer(
                language="es",
                backend="espeak",
                with_stress=True,
                preserve_punctuation=True,
            )
        elif g2p_type == "espeak_ng_russian":
            self.g2p = Phonemizer(
                language="ru",
                backend="espeak",
                with_stress=True,
                preserve_punctuation=True,
            )
        elif g2p_type == "espeak_ng_greek":
            self.g2p = Phonemizer(
                language="el",
                backend="espeak",
                with_stress=True,
                preserve_punctuation=True,
            )
        elif g2p_type == "espeak_ng_finnish":
            self.g2p = Phonemizer(
                language="fi",
                backend="espeak",
                with_stress=True,
                preserve_punctuation=True,
            )
        elif g2p_type == "espeak_ng_hungarian":
            self.g2p = Phonemizer(
                language="hu",
                backend="espeak",
                with_stress=True,
                preserve_punctuation=True,
            )
        elif g2p_type == "espeak_ng_dutch":
            self.g2p = Phonemizer(
                language="nl",
                backend="espeak",
                with_stress=True,
                preserve_punctuation=True,
            )
        elif g2p_type == "espeak_ng_hindi":
            self.g2p = Phonemizer(
                language="hi",
                backend="espeak",
                with_stress=True,
                preserve_punctuation=True,
            )
        elif g2p_type == "espeak_ng_italian":
            self.g2p = Phonemizer(
                language="it",
                backend="espeak",
                with_stress=True,
                preserve_punctuation=True,
            )
        elif g2p_type == "espeak_ng_polish":
            self.g2p = Phonemizer(
                language="pl",
                backend="espeak",
                with_stress=True,
                preserve_punctuation=True,
            )
        elif g2p_type == "g2pk":
            self.g2p = G2pk(no_space=False)
        elif g2p_type == "g2pk_no_space":
            self.g2p = G2pk(no_space=True)
        elif g2p_type == "g2pk_explicit_space":
            self.g2p = G2pk(explicit_space=True, space_symbol=space_symbol)
        elif g2p_type == "espeak_ng_english_us_vits":
            # VITS official implementation-like processing
            # Reference: https://github.com/jaywalnut310/vits
            self.g2p = Phonemizer(
                language="en-us",
                backend="espeak",
                with_stress=True,
                preserve_punctuation=True,
                strip=True,
                word_separator=" ",
                phone_separator="",
                split_by_single_token=True,
            )
        elif g2p_type == "korean_jaso":
            self.g2p = Jaso(space_symbol=space_symbol, no_space=False)
        elif g2p_type == "korean_jaso_no_space":
            self.g2p = Jaso(no_space=True)
        elif g2p_type == "g2p_is":
            self.g2p = IsG2p()
        elif g2p_type == "g2p_is_north":
            self.g2p = IsG2p(dialect="north")
        else:
            raise NotImplementedError(f"Not supported: g2p_type={g2p_type}")

        self.g2p_type = g2p_type
        self.space_symbol = space_symbol
        if non_linguistic_symbols is None:
            self.non_linguistic_symbols = set()
        elif isinstance(non_linguistic_symbols, (Path, str)):
            non_linguistic_symbols = Path(non_linguistic_symbols)
            try:
                with non_linguistic_symbols.open("r", encoding="utf-8") as f:
                    self.non_linguistic_symbols = set(line.rstrip() for line in f)
            except FileNotFoundError:
                warnings.warn(f"{non_linguistic_symbols} doesn't exist.")
                self.non_linguistic_symbols = set()
        else:
            self.non_linguistic_symbols = set(non_linguistic_symbols)
        self.remove_non_linguistic_symbols = remove_non_linguistic_symbols

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f'g2p_type="{self.g2p_type}", '
            f'space_symbol="{self.space_symbol}", '
            f'non_linguistic_symbols="{self.non_linguistic_symbols}"'
            ")"
        )

    def text2tokens(self, line: str) -> List[str]:
        tokens = []
        while len(line) != 0:
            for w in self.non_linguistic_symbols:
                if line.startswith(w):
                    if not self.remove_non_linguistic_symbols:
                        tokens.append(line[: len(w)])
                    line = line[len(w) :]
                    break
            else:
                t = line[0]
                tokens.append(t)
                line = line[1:]

        line = "".join(tokens)
        tokens = self.g2p(line)
        return tokens

    def tokens2text(self, tokens: Iterable[str]) -> str:
        # phoneme type is not invertible
        return "".join(tokens)

    def text2tokens_svs(self, syllable: str) -> List[str]:
        # Note(Yuning): fix syllabel2phoneme mismatch
        # If needed, customed_dic can be changed into extra input
        customed_dic = {
            "へ": ["h", "e"],
            "は": ["h", "a"],
            "シ": ["sh", "I"],
            "ヴぁ": ["v", "a"],
            "ヴぃ": ["v", "i"],
            "ヴぇ": ["v", "e"],
            "ヴぉ": ["v", "o"],
            "でぇ": ["dy", "e"],
            "くぁ": ["k", "w", "a"],
            "くぃ": ["k", "w", "i"],
            "くぅ": ["k", "w", "u"],
            "くぇ": ["k", "w", "e"],
            "くぉ": ["k", "w", "o"],
            "ぐぁ": ["g", "w", "a"],
            "ぐぃ": ["g", "w", "i"],
            "ぐぅ": ["g", "w", "u"],
            "ぐぇ": ["g", "w", "e"],
            "ぐぉ": ["g", "w", "o"],
            "くぉっ": ["k", "w", "o", "cl"],
        }
        tokens = self.g2p(syllable)
        if syllable in customed_dic:
            tokens = customed_dic[syllable]
        return tokens
