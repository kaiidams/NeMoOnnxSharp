namespace NeMoOnnxSharp
{
    // nemo.collections.tts.torch.g2ps.EnglishG2p
    class EnglishG2p
    {
        public string[] Process(string text)
        {
            return text.Split();
        }
    }

    abstract class BaseTokenizer
    {
        protected string Pad = "<pad>";
        protected string Blank = "<blank>";
        protected string OOV = "<oov>";

        protected BaseTokenizer()
        {
            _id2token = Array.Empty<string>();
        }

        protected string[] _id2token;
    }

    // nemo.collections.tts.torch.tts_tokenizers.EnglishPhonemesTokenizer
    class EnglishPhonemesTokenizer : BaseTokenizer
    {
        /// <summary>
        /// English phoneme-based tokenizer.
        /// Args:
        ///    g2p: Grapheme to phoneme module.
        ///    punct: Whether to reserve grapheme for basic punctuation or not.
        ///    non_default_punct_list: List of punctuation marks which will be used instead default.
        ///    stresses: Whether to use phonemes codes with stresses (0-2) or not.
        ///    chars: Whether to additionally use chars together with phonemes. It is useful if g2p module can return chars too.
        ///    space: Space token as string.
        ///    silence: Silence token as string (will be disabled if it is None).
        ///    apostrophe: Whether to use apostrophe or not.
        ///    oov: OOV token as string.
        ///    sep: Separation token as string.
        ///    add_blank_at: Add blank to labels in the specified order ("last") or after tokens (any non None),
        ///     if None then no blank in labels.
        ///    pad_with_space: Whether to pad text with spaces at the beginning and at the end or not.
        ///    text_preprocessing_func: Text preprocessing function for correct execution of the tokenizer.
        ///     Basically, it replaces all non-unicode characters with unicode ones.
        ///     Note that lower() function shouldn't be applied here, in case the text contains phonemes (it will be handled by g2p).
        /// </summary>
        /// <param name="g2p"></param>
        /// <param name="punct"></param>
        /// <param name="non_default_punct_list"></param>
        /// <param name="stresses"></param>
        /// <param name="chars"></param>
        /// <param name="space"></param>
        /// <param name="silence"></param>
        /// <param name="apostrophe"></param>
        /// <param name="oov"></param>
        /// <param name="sep"></param>
        /// <param name="add_blank_at"></param>
        /// <param name="pad_with_space"></param>
        public EnglishPhonemesTokenizer(
            EnglishG2p g2p,
            bool punct = true,
            object? non_default_punct_list = null,
            bool stresses = false,
            bool chars = false,
            string space = " ",
            string? silence = null,
            bool apostrophe = true,
            object? oov = null, // BaseTokenizer.OOV,
            char sep='|',  // To be able to distinguish between 2/3 letters codes.
            object? add_blank_at = null,
            bool pad_with_space=false)
            // object? text_preprocessing_func=lambda text: english_text_preprocessing(text, lower=false),
        {
            _phonemeProbability = null;
            _g2p = g2p;
            _space = 0;
            var tokens = new List<string>();
            tokens.Add(space);

            if (silence is not null)
            {
                throw new Exception();
            }

            tokens.AddRange(Consonants);
            var vowels = Vowels;

            if (stresses)
            {
                vowels = vowels.SelectMany(p => Enumerable.Range(0, 3), (p, s) => $"{p}{s}").ToArray();
            }
            tokens.AddRange(vowels);

            if (chars || _phonemeProbability is not null)
            {
                if (!chars)
                {
                    // logging.warning(
                    //     "phoneme_probability was not None, characters will be enabled even though "
                    //     "chars was set to False."
                    // );
                }
                tokens.AddRange(AsciiLowercase.Select(ch => ch.ToString()));
            }

            if (apostrophe)
            {
                tokens.Add("'");  // Apostrophe
            }

            if (punct)
            {
                if (non_default_punct_list is not null)
                {
                    throw new Exception();
                    // self.PunctList = non_default_punct_list;
                }
                tokens.AddRange(PunctList);
            }

            tokens.Add(Pad);  // Out Of Vocabulary
            _pad = tokens.Count;
            tokens.Add(Blank);  // Out Of Vocabulary
            _blank = tokens.Count;
            tokens.Add(OOV);  // Out Of Vocabulary
            _oov = tokens.Count;

            _id2token = tokens.ToArray();
            _token2id = new Dictionary<string, int>(
                Enumerable.Range(0, _id2token.Length)
                .Select(i => new KeyValuePair<string, int>(_id2token[i], i)));
        }

        public int[] Encode(string text)
        {
            var g2pText = _g2p.Process(text);
            var ps = new List<string>();
            foreach (var t in g2pText)
            {
                foreach (var c in t.ToLower())
                {
                    if (_token2id.ContainsKey(c.ToString()))
                        ps.Add(c.ToString());
                }

            }
            return EncodeFromG2p(ps.ToArray());
        }

        private int[] EncodeFromG2p(string[] g2pText)
        {
            return g2pText.Select(p => _token2id[p]).ToArray();
        }

        private readonly string[] PunctList =
        {  // Derived from LJSpeech and "/" additionally
            ",", ".", "!", "?", "-",
            ":", ";", "/", "\"", "(",
            ")", "[", "]", "{", "}",
        };
        private readonly string[] Vowels = {
            "AA", "AE", "AH", "AO", "AW",
            "AY", "EH", "ER", "EY", "IH",
            "IY", "OW", "OY", "UH", "UW",
        };
        private readonly string[] Consonants = {
            "B", "CH", "D", "DH", "F", "G",
            "HH", "JH", "K", "L", "M", "N",
            "NG", "P", "R", "S", "SH", "T",
            "TH", "V", "W", "Y", "Z", "ZH",
        };

        private const string AsciiLowercase = "abcdefghijklmnopqrstuvwxyz";

        private readonly EnglishG2p _g2p;
        private IDictionary<string, int> _token2id;
        private readonly int _space;
        private readonly int _pad;
        private readonly int _blank;
        private readonly int _oov;
        private readonly object? _phonemeProbability;
    }

    class Program
    {
        static void Main(string[] args)
        {
            var g2p = new EnglishG2p();
            // text_tokenizer:
            //     _target_: nemo.collections.tts.torch.tts_tokenizers.EnglishPhonemesTokenizer
            //     punct: true
            //     stresses: true
            //     chars: true
            //     apostrophe: true
            //     pad_with_space: true
            //     g2p:
            //         _target_: nemo.collections.tts.torch.g2ps.EnglishG2p
            //         phoneme_dict: ../../scripts/tts_dataset_files/cmudict-0.7b_nv22.10
            //         heteronyms: ../../scripts/tts_dataset_files/heteronyms-052722
            //         phoneme_probability: 0.5
            //     add_blank_at: true
            var tokenizer = new EnglishPhonemesTokenizer(
                g2p,
                punct: true,
                stresses: true,
                chars: true,
                apostrophe: true,
                pad_with_space: true,
                add_blank_at: true);
            var encoded = tokenizer.Encode("Hello World!");
            
            foreach (var x in encoded)
            {
                Console.WriteLine("{0}", x);
            }
            Console.WriteLine("end");
        }
    }
//     [' ',
//  'B',
//  'CH',
//  'D',
//  'DH',
//  'F',
//  'G',
//  'HH',
//  'JH',
//  'K',
//  'L',
//  'M',
//  'N',
//  'NG',
//  'P',
//  'R',
//  'S',
//  'SH',
//  'T',
//  'TH',
//  'V',
//  'W',
//  'Y',
//  'Z',
//  'ZH',
//  'AA0',
//  'AA1',
//  'AA2',
//  'AE0',
//  'AE1',
//  'AE2',
//  'AH0',
//  'AH1',
//  'AH2',
//  'AO0',
//  'AO1',
//  'AO2',
//  'AW0',
//  'AW1',
//  'AW2',
//  'AY0',
//  'AY1',
//  'AY2',
//  'EH0',
//  'EH1',
//  'EH2',
//  'ER0',
//  'ER1',
//  'ER2',
//  'EY0',
//  'EY1',
//  'EY2',
//  'IH0',
//  'IH1',
//  'IH2',
//  'IY0',
//  'IY1',
//  'IY2',
//  'OW0',
//  'OW1',
//  'OW2',
//  'OY0',
//  'OY1',
//  'OY2',
//  'UH0',
//  'UH1',
//  'UH2',
//  'UW0',
//  'UW1',
//  'UW2',
//  'a',
//  'b',
//  'c',
//  'd',
//  'e',
//  'f',
//  'g',
//  'h',
//  'i',
//  'j',
//  'k',
//  'l',
//  'm',
//  'n',
//  'o',
//  'p',
//  'q',
//  'r',
//  's',
//  't',
//  'u',
//  'v',
//  'w',
//  'x',
//  'y',
//  'z',
//  "'",
//  ',',
//  '.',
//  '!',
//  '?',
//  '-',
//  ':',
//  ';',
//  '/',
//  '"',
//  '(',
//  ')',
//  '[',
//  ']',
//  '{',
//  '}',
//  '<pad>',
//  '<blank>',
//  '<oov>']
}