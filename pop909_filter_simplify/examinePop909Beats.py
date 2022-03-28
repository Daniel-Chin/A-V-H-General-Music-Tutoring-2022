POP909PATH = Plase_supply_path_to_pop909_here

import os
import random
import mido
from matplotlib import pyplot as plt

class AbsoluteTimeMessage:
    def __init__(
        self, midoMessage : mido.Message, abs_time, 
        delta_time, 
    ):
        self.note = None
        for name in ['type', 'is_meta', 'note', 'velocity']:
            try:
                self.__setattr__(name, midoMessage.__getattribute__(name))
            except AttributeError:
                pass
        self.abs_time = abs_time
        self.delta_time = delta_time
        self.duration = None
    
    def __eq__(self, o: object) -> bool:
        try:
            return self.note == o.note
        except AttributeError:
            return False
    
    def __repr__(self) -> str:
        return f'<absMsg {self.type} note={self.note} time={self.abs_time}>'

def main():
    os.chdir(POP909PATH)
    list_dir = os.listdir()
    songs = [x for x in list_dir if os.path.isdir(x)]
    random.shuffle(songs)
    for song_id in songs:
        mid = mido.MidiFile(f'{song_id}/{song_id}.mid')
        absSong = absolutize(mid)
        sec_per_beat = getTempo(mid) / 10e5
        notes = set()
        for msg in absSong:
            if msg.type == 'note_on':
                plt.plot(
                    [msg.abs_time, msg.abs_time + msg.duration], 
                    [msg.note, msg.note], 
                    c='k', linewidth=2, 
                )
                notes.add(msg.note)
        plt.title(song_id)
        with open(f'{song_id}/beat_midi.txt', 'r') as f:
            for line in f:
                beat = float(line.strip().split(' ', 1)[0])
                plt.axvline(beat, c='r')
        st = sorted([*notes])
        _min, _max = st[0], st[-1]
        plt.axis([0, sec_per_beat * 40, _min, _max])
        plt.show()

def getTempo(mid):
    for msg in mid.tracks[0]:
        if msg.type == 'set_tempo':
            return msg.tempo
    raise Exception('set_tempo msg not found')

def second2tick(sec, mid):
    return round(mido.second2tick(
        sec, mid.ticks_per_beat, getTempo(mid), 
    ))

def absolutize(mid : mido.MidiFile):
    absSong = []
    acc_time = 0
    delta_time = 0
    unclosed = []
    for msg in mid.tracks[1]:
        time = mido.tick2second(
            msg.time, mid.ticks_per_beat, getTempo(mid), 
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

main()
