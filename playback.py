from music21 import midi
from mido import MidiFile
from pyo import *  # cursed but the docs and S.O. use this convention


def pyo_example(started_server):  # good for checking if any sound at all is coming out
    a = Sine(mul=0.01).out()
    started_server.gui(locals())


def init():
    return Server().boot().start()


def scale_stream_tempo(note_stream, tempo=0.5):
    return note_stream.augmentOrDiminish(tempo)


def stream_to_midi_file(note_stream):
    """writes music21 stream to temp midi file to be 'picked up' by other libs; returns the filepath"""
    fp = note_stream.write('midi', fp='midifile.mid')
    return fp


def synthesize_midi_test_1(started_server, midi_file_path, pan_ratio, output_filename="test.wav"):

    print(midi_file_path)
    # ALSO not necessary later, just a little audio synth to play the MIDI events:
    # A little audio synth to play the MIDI events.
    mid = Notein()

    amp = MidiAdsr(mid["velocity"])
    pit = MToF(mid["pitch"])
    osc = Osc(SquareTable(), freq=pit, mul=amp).mix(1)

    rev = STRev(osc, revtime=1, cutoff=4000, bal=0.2)
    p = Pan(rev, pan=pan_ratio, spread=0.05)
    type(p)
    rec = Record(p, filename=output_filename, fileformat=0, sampletype=4)
    p.out()

    # Opening the MIDI file...
    mid = MidiFile(midi_file_path)

    # ... and reading its content.
    for message in mid.play():
        started_server.addMidiEvent(*message.bytes())

    rec.stop()
