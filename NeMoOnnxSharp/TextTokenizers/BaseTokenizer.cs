// Copyright (c) Katsuya Iida.  All Rights Reserved.
// See LICENSE in the project root for license information.

// A number of implementation details in this file have been translated from the Python scripts of NVIDIA NeMo,
// largely located in the files found in this folder:
//
// https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/tts/torch/tts_tokenizers.py
//
// The origin has the following copyright notice and license:
//
// https://github.com/NVIDIA/NeMo/blob/main/LICENSE
//

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeMoOnnxSharp.TextTokenizers
{
    public abstract class BaseTokenizer
    {
        public enum AddBlankAt
        {
            False,
            True,
            Last
        }

        protected const string Pad = "<pad>";
        protected const string Blank = "<blank>";
        protected const string OOV = "<oov>";

        protected BaseTokenizer()
        {
            _sep = string.Empty;
            _id2token = Array.Empty<string>();
            _token2id = new Dictionary<string, int>();
            _utilIds = new HashSet<int>();
        }

        /// <summary>
        /// Turns str text into int tokens.
        /// </summary>
        public abstract int[] Encode(string text);

        /// <summary>
        /// Turns ints tokens into str text.
        /// </summary>
        public string Decode(int[] tokens)
        {
            return string.Join(
                _sep,
                tokens
                .Where(t => !_utilIds.Contains(t))
                .Select(t => _id2token[t]));
        }

        public string[] Tokens { get { return _id2token; } }
        public int PadId { get { return _pad; } }
        public int BlankId { get { return _blank; } }
        public int OOVId { get { return _oov; } }
        public string Sep { get { return _sep; } }

        protected string[] _id2token;
        protected IDictionary<string, int> _token2id;
        protected ISet<int> _utilIds;
        protected int _space;
        protected int _pad;
        protected int _blank;
        protected int _oov;
        protected string _sep;
        protected bool _padWithSpace;
    }
}
