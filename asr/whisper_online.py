"""
Whisper online implementation.
# REFERENCE : https://github.com/ufal/whisper_streaming
"""

#!/usr/bin/env python3
import sys
import numpy as np
import logging

class HypothesisBuffer:

    def __init__(self, logfile=sys.stderr):
        self.commited_in_buffer = []
        self.buffer = []
        self.new = []

        self.last_commited_time = 0
        self.last_commited_word = None

        self.logfile = logfile

    def insert(self, new, offset):
        # compare self.commited_in_buffer and new. It inserts only the
        # words in new that extend the commited_in_buffer, it means
        # they are roughly behind last_commited_time and new in content
        # the new tail is added to self.new

        new = [(a + offset, b + offset, t) for a, b, t in new]
        self.new = [(a, b, t) for a, b, t in new
                    if a > self.last_commited_time - 0.1]

        if len(self.new) >= 1:
            a, b, t = self.new[0]
            if abs(a - self.last_commited_time) < 1:
                if self.commited_in_buffer:
                    # it's going to search for 1, 2, ..., 5
                    # consecutive words (n-grams) that are identical
                    # in commited and new. If they are, they're dropped.
                    cn = len(self.commited_in_buffer)
                    nn = len(self.new)
                    for i in range(1, min(cn, nn, 5) + 1):
                        # 5 is the maximum
                        c = " ".join(
                            [self.commited_in_buffer[-j][2]
                             for j in range(1, i + 1)][::-1]
                        )
                        tail = " ".join(self.new[j - 1][2]
                                        for j in range(1, i + 1))
                        if c == tail:
                            logging.info("removing last %s words:", i)
                            for j in range(i):
                                logging.info("\t %s",self.new.pop(0))
                            break

    def flush(self):
        # returns commited chunk = the longest common
        # prefix of 2 last inserts.

        commit = []
        while self.new:
            na, nb, nt = self.new[0]

            if len(self.buffer) == 0:
                break

            if nt == self.buffer[0][2]:
                commit.append((na, nb, nt))
                self.last_commited_word = nt
                self.last_commited_time = nb
                self.buffer.pop(0)
                self.new.pop(0)
            else:
                break
        self.buffer = self.new
        self.new = []
        self.commited_in_buffer.extend(commit)
        return commit

    def pop_commited(self, time):
        while self.commited_in_buffer and \
                self.commited_in_buffer[0][1] <= time:
            self.commited_in_buffer.pop(0)

    def complete(self):
        return self.buffer


class OnlineASRProcessor:

    SAMPLING_RATE = 16000

    def __init__(self, asr, tokenizer=None, buffer_trimming=("segment", 15),
                 logfile=sys.stderr):
        """
        asr: WhisperASR object
        tokenizer: sentence tokenizer object for the target language.
                   Must have a method *split* that behaves like the one
                   of MosesTokenizer. It can be None, if "segment" buffer
                   trimming option is used, then tokenizer is not used at all.
                   ("segment", 15)
        buffer_trimming: a pair of (option, seconds), where option is either
                        "sentence" or "segment", and seconds is a number.
                        Buffer is trimmed if it is longer than "seconds"
                        threshold. Default is the most recommended option.
        logfile: where to store the log.
        """
        self.asr = asr
        self.tokenizer = tokenizer
        self.logfile = logfile
        self.audio_buffer = None
        self.buffer_time_offset = 0
        self.last_chunked_at = 0
        self.init()

        self.buffer_trimming_way, self.buffer_trimming_sec = buffer_trimming

    def init(self):
        """run this when starting or restarting processing"""
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_time_offset = 0

        self.transcript_buffer = HypothesisBuffer(logfile=self.logfile)
        self.commited = []
        self.last_chunked_at = 0

        self.silence_iters = 0

    def insert_audio_chunk(self, audio):
        self.audio_buffer = np.append(self.audio_buffer, audio)

    def prompt(self):
        """
        Returns a tuple: (prompt, context), where "prompt" is a 200-character
        suffix of commited text that is inside of the scrolled away part of
        audio buffer.
        "context" is the commited text that is inside the audio buffer.
        It is transcribed again and skipped. It is returned only for debugging
        and logging reasons.
        """
        k = max(0, len(self.commited) - 1)
        while k > 0 and self.commited[k - 1][1] > self.last_chunked_at:
            k -= 1

        p = self.commited[:k]
        p = [t for _, _, t in p]
        prompt = []
        length = 0
        while p and length < 200:  # 200 characters prompt size
            x = p.pop(-1)
            length += len(x) + 1
            prompt.append(x)
        non_prompt = self.commited[k:]
        return self.asr.sep.join(prompt[::-1]), self.asr.sep.join(
            t for _, _, t in non_prompt
        )

    def process_iter(self):
        """
        Runs on the current audio buffer.
        Returns: a tuple (beg_timestamp, end_timestamp, "text")
                 or (None, None, "").
        The non-emty text is confirmed (committed) partial transcript.

        """
        prompt, _ = self.prompt()
        # print("PROMPT:", prompt, file=self.logfile)
        # print("CONTEXT:", non_prompt, file=self.logfile)
        # print(f"transcribing {len(self.audio_buffer)/\
        # self.SAMPLING_RATE:2.2f} seconds from \
        # {self.buffer_time_offset:2.2f}",file=self.logfile)
        res = self.asr(self.audio_buffer, init_prompt=prompt)

        # transform to [(beg,end,"word1"), ...]
        tsw = self.asr.ts_words(res)

        self.transcript_buffer.insert(tsw, self.buffer_time_offset)
        o = self.transcript_buffer.flush()
        self.commited.extend(o)
        # print(">>>>COMPLETE NOW:",self.to_flush(o),\
        # file=self.logfile,flush=True)
        # print("INCOMPLETE:",\
        # self.to_flush(self.transcript_buffer.complete()),\
        # file=self.logfile,flush=True)

        # there is a newly confirmed text

        # trim the completed sentences
        if o and self.buffer_trimming_way == "sentence":
            if (
                len(self.audio_buffer) / self.SAMPLING_RATE >
                self.buffer_trimming_sec
            ):  # longer than this
                self.chunk_completed_sentence()

        if self.buffer_trimming_way == "segment":
            # trim the completed segments longer than s
            s = self.buffer_trimming_sec
        else:
            s = 30  # if the audio buffer is longer than 30s, trim it

        if len(self.audio_buffer) / self.SAMPLING_RATE > s:
            self.chunk_completed_segment(res)

            # alternative: on any word
            # l = self.buffer_time_offset +
            # len(self.audio_buffer)/self.SAMPLING_RATE - 10
            # let's find commited word that is less
            # k = len(self.commited)-1
            # while k>0 and self.commited[k][1] > l:
            #    k -= 1
            # t = self.commited[k][1]
            logging.info("chunking segment")
            # self.chunk_at(t)

        logging.info(
            "len of buffer now: %s",
            len(self.audio_buffer)/self.SAMPLING_RATE
        )
        return self.to_flush(o)

    def chunk_completed_sentence(self):
        if self.commited == []:
            return
        logging.info(self.commited)
        sents = self.words_to_sentences(self.commited)
        for s in sents:
            logging.info("\t\tSENT:{%s}",s)
        if len(sents) < 2:
            return
        while len(sents) > 2:
            sents.pop(0)
        # we will continue with audio processing at this timestamp
        chunk_at = sents[-2][1]

        logging.info("--- sentence chunked at %s",chunk_at)
        self.chunk_at(chunk_at)

    def chunk_completed_segment(self, res):
        if self.commited == []:
            return

        ends = self.asr.segments_end_ts(res)

        t = self.commited[-1][1]

        if len(ends) > 1:

            e = ends[-2] + self.buffer_time_offset
            while len(ends) > 2 and e > t:
                ends.pop(-1)
                e = ends[-2] + self.buffer_time_offset
            if e <= t:
                logging.info("--- segment chunked at %s",e)
                self.chunk_at(e)
            else:
                logging.info("--- last segment not within commited area")
        else:
            logging.info("--- not enough segments to chunk")

    def chunk_at(self, time):
        """trims the hypothesis and audio buffer at "time" """
        self.transcript_buffer.pop_commited(time)
        cut_seconds = time - self.buffer_time_offset
        self.audio_buffer = self.audio_buffer[
                            int(cut_seconds * self.SAMPLING_RATE):]
        self.buffer_time_offset = time
        self.last_chunked_at = time

    def words_to_sentences(self, words):
        """Uses self.tokenizer for sentence segmentation of words.
        Returns: [(beg,end,"sentence 1"),...]
        """

        cwords = [w for w in words]
        t = " ".join(o[2] for o in cwords)
        s = self.tokenizer.split(t)
        out = []
        while s:
            beg = None
            end = None
            sent = s.pop(0).strip()
            fsent = sent
            while cwords:
                b, e, w = cwords.pop(0)
                w = w.strip()
                if beg is None and sent.startswith(w):
                    beg = b
                elif end is None and sent == w:
                    end = e
                    out.append((beg, end, fsent))
                    break
                sent = sent[len(w):].strip()
        return out

    def finish(self):
        """Flush the incomplete text when the whole processing ends.
        Returns: the same format as self.process_iter()
        """
        o = self.transcript_buffer.complete()
        f = self.to_flush(o)
        logging.info("last, noncommited: %s",f)
        return f

    def to_flush(
        self,
        sents,
        sep=None,
        offset=0,
    ):
        # concatenates the timestamped words or sentences
        # into one sequence that is flushed in one line
        # sents: [(beg1, end1, "sentence1"), ...] or [] if empty
        # return: (beg1,end-of-last-sentence,"concatenation of sentences")
        # or (None, None, "") if empty
        if sep is None:
            sep = self.asr.sep
        t = sep.join(s[2] for s in sents)
        if len(sents) == 0:
            b = None
            e = None
        else:
            b = offset + sents[0][0]
            e = offset + sents[-1][1]
        return (b, e, t)
