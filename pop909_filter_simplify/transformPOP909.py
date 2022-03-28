Plase_supply_path_to_pop909_here
POP909PATH = '.../pop909/POP909'
DAISHUPATH = '.../DaishuPOP909/POP909'

DAISHU = 'DAISHU'
ORIGINAL = 'ORIGINAL'
PADDING = .1       # otherwise in Voldemort visuals, consecutive same-pitch notes would merge
SPLIT_THRESHOLD = 5
NOTE_OFF_QUANTIZE = .5        # how many beats
USING = DAISHU
N_QUAVERS_P_B = 4

INVALID_Q_DURATION = (5, 7, 9, 10, 11, 13, 14, 15)

import os
from typing import Tuple, List
import csv
from sympy.ntheory import factorint
import math
import mido
try:
    from jdt import Jdt
    from editDistance import editDistance
except ImportError as e:
    module_name = str(e).split('No module named ', 1)[1].strip().strip('"\'')
    if module_name in (
        'jdt', 'editDistance', 
    ):
        print(f'Missing module {module_name}. Please download at')
        print(f'https://github.com/Daniel-Chin/Python_Lib')
        input('Press Enter to quit...')
    raise e

class AbsoluteTimeMessage:
    __slot__ = [
        'note', 'abs_time', 'delta_time', 'type', 'is_meta', 
        'velocity', 'duration', 'n_beam', 'is_dotted', 
        'is_hollow', 'q_time', 'q_duration', 
    ]

    def __init__(
        self, midoMessage : mido.Message, abs_time, 
        delta_time, 
    ):
        self.note = None
        if midoMessage is not None:
            for name in ['type', 'is_meta', 'note', 'velocity']:
                try:
                    self.__setattr__(name, midoMessage.__getattribute__(name))
                except AttributeError:
                    pass
        self.abs_time = abs_time
        self.delta_time = delta_time
        self.duration = None
        self.n_beam = None
        self.is_dotted = False
        self.is_hollow = False
        self.q_time = None
        self.q_duration = None
    
    def __eq__(self, o: object) -> bool:
        try:
            return self.note == o.note
        except AttributeError:
            return False
    
    def __repr__(self) -> str:
        return f'<absMsg {self.type} note={self.note} time={self.abs_time}>'
    
    def copy(self):
        msg = AbsoluteTimeMessage(None, self.abs_time, self.delta_time)
        for name in self.__slot__:
            msg.__setattr__(name, self.__getattribute__(name))
        return msg

def main():
    os.chdir(POP909PATH)
    list_dir = os.listdir()
    songs = [x for x in list_dir if os.path.isdir(x)]
    meta = {}
    try:
        with open('index.csv', 'r', encoding='utf-8') as f:
            c = csv.DictReader(f)
            for line in c:
                meta[line['song_id']] = line
    except FileNotFoundError as e:
        print('Convert XLSX to CSV first!')
        raise e
    results = []
    with Jdt(len(songs)) as j:
        for song_id in songs:
            if USING is ORIGINAL:
                mid = mido.MidiFile(f'{song_id}/{song_id}.mid')
            else:
                mid = None
            sec_per_beat = getTempo(mid, song_id) / 10e5
            num_beats_per_measure = int(
                meta[song_id]['num_beats_per_measure']
            )
            absSong = absolutize(mid, j, song_id, sec_per_beat)
            if USING is ORIGINAL:
                beats_time = []
                is_downbeat = []
                with open(f'{song_id}/beat_midi.txt', 'r') as f:
                    for line in f:
                        time, bo, boe = line.strip().split(' ')
                        beats_time.append(float(time))
                        if boe == '1.0':
                            is_downbeat.append(True)
                        elif boe == '0.0': 
                            is_downbeat.append(False)
                        else:
                            raise ValueError(f'boe = "{boe}"')
                straightenBeats(
                    absSong, song_id, sec_per_beat, j, 
                    beats_time, 
                )
            elif USING is DAISHU:
                is_downbeat = None
            segments = split(
                absSong, num_beats_per_measure, 
                is_downbeat, sec_per_beat, 
            )
            shifted_segments = []
            for segment in segments:
                is_valid, loss = filterAndShift(segment)
                if is_valid and loss == 0:
                    shifted_segments.append(segment)
            if shifted_segments:
                tunes = removeDuplicate(shifted_segments, j)
                qTunes = []
                for tune in tunes:
                    removeUnichords(tune)
                    qTunes.append(quantize(tune, sec_per_beat, num_beats_per_measure))
                results.append((song_id, qTunes, sec_per_beat, num_beats_per_measure))
            j.acc()
    print('Preparing disk...')
    os.chdir(os.path.dirname(__file__))
    os.chdir('output')
    print('deleting previous files...')
    os.system('del .\\*.csv')
    print('writing to disk...')
    writeCsvs(results, meta)
    print('ok')

def getTempo(mid, song_id):
    if USING is ORIGINAL:
        for msg in mid.tracks[0]:
            if msg.type == 'set_tempo':
                return msg.tempo
        raise Exception('set_tempo msg not found')
    elif USING is DAISHU:
        with open(os.path.join(DAISHUPATH, song_id, 'tempo.txt'), 'r') as f:
            bpm = int(f.read().strip())
        sec_per_beat = 60 / bpm
        return sec_per_beat * 10e5

def absolutize(mid : mido.MidiFile, j : Jdt, song_id, sec_per_beat):
    if USING is DAISHU:
        notes = []
        with open(os.path.join(DAISHUPATH, song_id, 'melody.txt'), 'r') as f:
            for line in f:
                pitch, n_quavers = line.strip().split(' ')
                notes.append((int(pitch), int(n_quavers)))
        absSong = []
        acc_time = 0
        delta_time = 0
        for pitch, n_quavers in notes:
            duration = n_quavers / N_QUAVERS_P_B * sec_per_beat
            if pitch != 0:
                absMsg = AbsoluteTimeMessage(
                    None, acc_time, delta_time, 
                )
                delta_time = 0
                absMsg.duration = duration
                absMsg.type = 'note_on'
                absMsg.is_meta = False
                absMsg.note = pitch
                absMsg.velocity = 60
                absSong.append(absMsg)
            delta_time += duration
            acc_time += duration
    elif USING is ORIGINAL:
        assert mid.tracks[0][1].is_meta
        assert mid.tracks[1][0].name == 'MELODY'
        absSong = []
        acc_time = 0
        delta_time = 0
        unclosed = []
        for msg in mid.tracks[1]:
            time = mido.tick2second(
                msg.time, mid.ticks_per_beat, getTempo(mid, song_id), 
            )
            delta_time += time
            if msg.type == 'note_on' and msg.velocity == 0:
                for i, absMsg in enumerate(unclosed):
                    if absMsg.note == msg.note:
                        absMsg.duration = time
                        unclosed.pop(i)
                        if i != 0:
                            j.print('out-of-order note_off')
                        break
                else:
                    raise Exception
            else:
                acc_time += delta_time
                absMsg = AbsoluteTimeMessage(
                    msg, acc_time, delta_time, 
                )
                delta_time = 0
                absSong.append(absMsg)
                if absMsg.type == 'note_on':
                    unclosed.append(absMsg)
        assert not unclosed
    return absSong

def filterAndShift(
    absSong : List[AbsoluteTimeMessage], 
) -> Tuple[bool, int]:
    loss = 0
    pitches = set()
    for msg in absSong:
        if msg.type == 'note_on':
            pitches.add(msg.note)
    pitch_classes = set([x % 12 for x in pitches])
    sorted_pitches : list = sorted(pitches)
    candidates : List[Tuple[int, int]] = []
    for transpose in [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6]:
        if isDiatone([
            (x + transpose) % 12 for x in pitch_classes
        ]):
            while sorted_pitches[0] + transpose < 65:
                transpose += 12
            for pitch in reversed(sorted_pitches):
                if pitch + transpose > 83:
                    loss += 1
                else:
                    break
            candidates.append((transpose, loss))
        else:
            continue
    if not candidates:
        return False, None
    candidates.sort(key = lambda x : x[1])
    winner = candidates[0]
    for msg in absSong:
        if msg.type in ('note_on', 'note_off'):
            msg.note += winner[0]
    return True, winner[1]

def isDiatone(pitches):
    return all([x in (0, 2, 4, 5, 7, 9, 11) for x in pitches])

def straightenBeats(
    absSong : List[AbsoluteTimeMessage], song_id, sec_per_beat, 
    j : Jdt, beats, 
) -> List[AbsoluteTimeMessage]:
    beat_i = 0
    for msg in absSong:
        n_app_beats = 0
        while beats[1] < msg.abs_time:
            if len(beats) == 2:
                beats.append(beats[-1] * 2 - beats[-2])
                n_app_beats += 1
            beats.pop(0)
            beat_i += 1
        if n_app_beats:
            j.print('appending', n_app_beats, 'beats')
        msg.abs_time = ((msg.abs_time - beats[0]) / (
            beats[1] - beats[0]
        ) + beat_i) * sec_per_beat

def split(
    absSong : List[AbsoluteTimeMessage], 
    num_beats_per_measure, is_down_beat, sec_per_beat, 
):
    if USING is ORIGINAL:
        def isDownBeat(x):
            return is_down_beat[x]
    elif USING is DAISHU:
        def isDownBeat(x):
            return x % num_beats_per_measure == 0
    segments : List[List[AbsoluteTimeMessage]] = []
    last_cut = 0
    for i, msg in enumerate(absSong):
        if msg.delta_time >= SPLIT_THRESHOLD:
            segments.append(absSong[last_cut:i])
            last_cut = i
    segments.append(absSong[last_cut:])
    non_empty : List[List[AbsoluteTimeMessage]] = [
        segment for segment in segments 
        if any([msg.type == 'note_on' for msg in segment])
    ]
    for segment in non_empty:
        beat_i = int(segment[0].abs_time / sec_per_beat)
        try:
            while not isDownBeat(beat_i):
                beat_i -= 1
        except IndexError:
            beat_i = 0
            while not isDownBeat(beat_i):
                beat_i += 1
            beat_i -= num_beats_per_measure
        origin = beat_i * sec_per_beat
        for msg in segment:
            msg.abs_time -= origin
    return non_empty

def removeDuplicate(
    segments : List[List[AbsoluteTimeMessage]], j : Jdt, 
) -> List[List[AbsoluteTimeMessage]]:
    results = []
    while segments:
        one = segments.pop(0)
        for other in segments:
            edit_ratio = editDistance(one, other) / len(one)
            if edit_ratio < .2:
                j.print(
                    'remove duplicate. overlap', 
                    round((1 - edit_ratio) * 100), '%', 
                )
                break
        else:
            results.append(one)
    return results

def removeUnichords(tune):
    lastMsg = None
    for i, msg in reversed([*enumerate(tune)]):
        if lastMsg is not None:
            if lastMsg.abs_time == msg.abs_time and lastMsg.note == msg.note:
                print('Removing a pair of simultaneous same-pitch notes!')
                tune.pop(i)
        lastMsg = msg

def quantize(tune : List[AbsoluteTimeMessage], sec_per_beat, num_beats_per_measure):
    table = []
    todo = [msg for msg in tune if not msg.is_meta and msg.note is not None]
    division = None
    max_div = 0
    while todo:
        division = (division or 2) * 2
        if division >= 64:
            print('Multiple notes start at the same time!')
            [print(x) for x in todo]
            break
        max_div = max(max_div, division)
        sec_per_div = sec_per_beat / division
        ruler = {}
        table.append((sec_per_div, ruler))
        for msg in todo:
            q_time = round(msg.abs_time / sec_per_div)
            if q_time not in ruler:
                ruler[q_time] = []
            ruler[q_time].append(msg)
        todo.clear()
        for q_time in list(ruler):
            try:
                (ruler[q_time], ) = ruler[q_time]
            except ValueError:  # too many values to unpack
                todo.extend(ruler[q_time])
                ruler.pop(q_time)
    if max_div != 4:
        print('dividing a beat into', max_div)
        assert False

    min_sec_per_div = sec_per_beat / max_div

    # quantized repr
    tune = [msg for msg in tune if not msg.is_meta and msg.note is not None]
    for sec_per_div, ruler in table:
        for q_time, msg in ruler.items():
            msg: AbsoluteTimeMessage
            msg.q_time = q_time * round(sec_per_div / min_sec_per_div)
            msg.q_duration = round(msg.duration / min_sec_per_div)
            if msg.q_duration == 0:
                msg.q_duration = 1
    
    assert not any([x.q_time is None for x in tune])

    # insert rests
    newTune = []
    acc = 0
    for msg in tune:
        if msg.q_time > acc:
            rest = newRest()
            rest.q_time = acc
            rest.q_duration = msg.q_time - acc
            newTune.append(rest)
            acc = msg.q_time
        newTune.append(msg)
        acc += msg.q_duration
    n_mindivs_per_measure = num_beats_per_measure * max_div
    residual = (- acc) % n_mindivs_per_measure
    if residual:
        rest = newRest()
        rest.q_time = acc
        rest.q_duration = residual
        newTune.append(rest)
    tune = newTune

    assertContinuity(tune)

    # break cross-measure notes into two
    newTune = []
    for msg in tune:
        this_measure_end = int(1 + msg.q_time / n_mindivs_per_measure) * n_mindivs_per_measure
        while True:
            newTune.append(msg)
            time_left = this_measure_end - msg.q_time
            if msg.q_duration <= time_left:
                break
            newMsg = msg.copy()
            msg.q_duration = time_left
            msg = newMsg
            msg.q_time += time_left
            msg.q_duration -= time_left
            this_measure_end += n_mindivs_per_measure
    tune = newTune

    assertContinuity(tune)

    # quantize rest_on and rest_off to eighth
    newTune = []
    for i, msg in enumerate(tune):
        if msg.note == -1:
            if msg.q_time % 2 != 0:
                msg.q_time += 1
                msg.q_duration -= 1
                newTune[-1].q_duration += 1
            if msg.q_duration % 2 != 0:
                msg.q_duration -= 1
                tune[i + 1].q_time -= 1
                tune[i + 1].q_duration += 1
            if msg.q_duration != 0:
                newTune.append(msg)
        else:
            newTune.append(msg)
    tune = newTune
    
    assertContinuity(tune)

    # find {5,7,9,10,11,13,14,15}/4 - beat notes
    newTune = []
    for i, msg in enumerate(tune):
        if msg.q_duration in INVALID_Q_DURATION:
            try:
                nextMsg = tune[i + 1]
            except IndexError:
                next_is_rest = False
            else:
                next_is_rest = nextMsg.note == -1
            if next_is_rest:
                # modify next rest
                for cut in range(2, 16, 2):
                    if msg.q_duration - cut not in INVALID_Q_DURATION:
                        break
                msg.q_duration -= cut
                nextMsg.q_time -= cut
                nextMsg.q_duration += cut
                newTune.append(msg)
            else:
                # recursively subdivide
                newTune.extend(breakInvalid(msg))
        else:
            newTune.append(msg)
    tune = newTune

    assertContinuity(tune)

    # annotate note duration type
    for msg in tune:
        factors = twoThree(msg)
        if 3 in factors:
            msg.is_dotted = True
            factors[2] += 1
        msg.n_beam = round(math.log2(max_div)) - factors[2]
        if msg.n_beam == -1:
            msg.is_hollow = True
            msg.n_beam = 0
        assert msg.n_beam >= -1

    # Connect the beams
    beat_i = 0
    for i, msg in enumerate(tune):
        try:
            nextMsg = tune[i + 1]
        except IndexError:
            should_terminate = True
        else:
            should_terminate = False
            if nextMsg.n_beam == 0 or nextMsg.note == -1:
                should_terminate = True
            new_beat_i = nextMsg.q_time // max_div
            if new_beat_i != beat_i:
                beat_i = new_beat_i
                should_terminate = True
            if msg.note == -1:
                should_terminate = True
        if should_terminate:
            msg.n_beam *= -1
    
    # calculate real numbers
    for msg in tune:
        msg.abs_time = msg.q_time * min_sec_per_div
        msg.duration = msg.q_duration * min_sec_per_div - PADDING
    
    return tune

def newRest():
    rest = AbsoluteTimeMessage(None, None, None)
    rest.type = 'note_on'
    rest.is_meta = False
    rest.note = -1
    rest.velocity = 0
    return rest

def assertContinuity(tune):
    acc = 0
    for msg in tune:
        assert msg.q_time == acc
        acc += msg.q_duration

def breakInvalid(msg, by = 8):
    mid = round(1 + msg.q_time / by) * by
    q_noteoff = msg.q_time + msg.q_duration
    if mid < q_noteoff:
        left = msg.copy()
        left.q_duration = mid - msg.q_time
        right = msg
        right.q_duration = q_noteoff - mid
        right.q_time = mid
        if left.q_duration in INVALID_Q_DURATION:
            left = breakInvalid(left, by // 2)
        else:
            left = [left]
        if right.q_duration in INVALID_Q_DURATION:
            right = breakInvalid(right, by // 2)
        else:
            right = [right]
        return left + right
    return breakInvalid(msg, by // 2)

def twoThree(msg : AbsoluteTimeMessage):
    def ok(factors):
        for key in factors:
            if key not in (2, 3):
                return False
        if 3 in factors:
            return factors[3] == 1
        return True
    factors = factorint(msg.q_duration)
    if 2 not in factors:
        factors[2] = 0
    if ok(factors):
        return factors
    else:
        assert False

def writeCsvs(
    songs : List[Tuple[
        str, List[List[AbsoluteTimeMessage]], float, int, 
    ]], 
    meta, 
):
    with open(
        'index.csv', 'w', encoding='utf-8', newline='', 
    ) as rootF:
        rootCsv = csv.DictWriter(rootF, ['filename', 'title'])
        rootCsv.writeheader()
        for (
            song_id, tunes, sec_per_beat, 
            num_beats_per_measure, 
        ) in songs:
            for i, absSong in enumerate(tunes):
                filename = f'{song_id}_{i}.csv'
                rootCsv.writerow({
                    'filename': filename, 
                    'title': f'{meta[song_id]["name"]} sec {i}', 
                })
                with open(
                    filename, 'w', encoding='utf-8', newline='', 
                ) as f:
                    c = csv.writer(f)
                    writeOneCsv(absSong, c, sec_per_beat, num_beats_per_measure)

N_COLS = 8
def writePartialLine(c, partial):
    line = [''] * N_COLS
    line[:len(partial)] = partial
    c.writerow(line)

def writeOneCsv(
    absSong : List[AbsoluteTimeMessage], c, 
    sec_per_beat : float, 
    num_beats_per_measure : int, 
):
    writePartialLine(c, ['//', 'Define constants'])
    writePartialLine(c, ['prelude', 0, '// how many seconds after bgm plays should Movement start? This makes space for the leading beats. '])
    writePartialLine(c, ['default_transpose', 0])
    writePartialLine(c, ['measure_time', sec_per_beat * num_beats_per_measure])
    writePartialLine(c, ['metronome_per_measure', num_beats_per_measure])
    writePartialLine(c, ['//', 'Declare objects'])
    c.writerow([
        'class', 'pitch', 'velocity', 'note on', 'note off', 
        'beam count, negative means group termination, slash means no stem (whole note)', 'dotted?', 
        'hollow?', 
    ])
    for msg in absSong:
        if msg.type == 'note_on':
            if msg.duration is None:
                print('None duration at t =', msg.abs_time)
                msg.duration = .5
            c.writerow([
                'note', msg.note, msg.velocity, 
                msg.abs_time, msg.abs_time + msg.duration, 
                msg.n_beam, 
                't' if msg.is_dotted else '', 
                't' if msg.is_hollow else '', 
            ])

main()
