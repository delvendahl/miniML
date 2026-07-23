"""
Heka Patchmaster .dat file reader 
Adapted from https://github.com/campagnola/heka_reader

Structure definitions adapted from StimFit hekalib.cpp

Brief example::

    # Load a .dat file
    bundle = Bundle(file_name)
    
    # Select a trace
    trace = bundle.pul[group_ind][series_ind][sweep_ind][trace_ind]
    
    # Print meta-data for this trace
    print(trace)
    
    # Load data for this trace
    data = bundle.data[group_id, series_id, sweep_ind, trace_ind]

"""

import numpy as np
import re, struct, collections
import datetime


def cstr(byt):
    """Convert C string bytes to python string.
    """
    try:
        ind = byt.index(b'\0')
    except ValueError:
        return byt
    return byt[:ind].decode('utf-8', errors='ignore')


def cbyte(byt):
    """Convert C string byte to python integer.
    """
    try:
        byt[0]
    except ValueError:
        return byt
    return byt[0]


def cchar(byt):
    """Convert C char byte to python string.
    """
    return byt.decode('utf-8', errors='ignore')


def heka_time_to_date(time):
    time -= 1580970496 # JanFirst1990
    if time < 0:
        time += 4294967296 # HIGH_DWORD
    time += 9561652096 # MAC_BASE

    ref = datetime.datetime(1601, 1, 1) # Windows reference date
    conv_time = datetime.timedelta(seconds=time)

    return (ref + conv_time).strftime("%d-%b-%Y %H:%M:%S.%f")


def getFromList(lst, index):

    try:
        return lst[index]
    except IndexError:
        return f"Unknown (value: {index})"


def getAmplifierType(byte):
    return getFromList(["EPC7", "EPC8", "EPC9", "EPC10", "EPC10Plus"], byte)


def getADBoard(byte):
    return getFromList(["ITC16", "ITC18", "LIH1600", "LIH 8+8"], byte)


def getRecordingMode(byte):
    return getFromList(["InOut", "OnCell", "OutOut", "WholeCell", "CClamp", "VClamp", "NoMode"], byte)


def getDataFormat(byte):
    return getFromList(["int16", "int32", "real32", "real64"], byte)


def getSegmentClass(byte):
    return getFromList(["Constant", "Ramp", "Continuous", "ConstSine", "Squarewave", "Chirpwave"], byte)


def getStoreType(byte):
    return getFromList(["NoStore", "Store", "StoreStart", "StoreEnd"], byte)


def getIncrementMode(byte):
    return getFromList(["Inc", "Dec", "IncInterleaved", "DecInterleaved",
                        "Alternate", "LogInc", "LogDec", "LogIncInterleaved",
                        "LogDecInterleaved", "LogAlternate", "Toggle"], byte)


def getSourceType(byte):
    return getFromList(["Constant", "Hold", "Parameter"], byte)


def getAmplifierGain(byte):
    """
    Units: V/A
    """

    # Original units: mV/pA
    return getFromList([1e-3/1e-12 * x for x in
                       [0.005, 0.010, 0.020, 0.050, 0.1, 0.2,
                        0.5, 1, 2, 5, 10, 20,
                        50, 100, 200, 500, 1000, 2000]], byte)


def convertDataFormatToNP(dataFormat):

    d = {"int16": np.int16,
         "int32": np.int32,
         "real32": np.float32,
         "real64": np.float64}

    return d[dataFormat]


def getClampMode(byte):
    return getFromList(["TestMode", "VCMode", "CCMode", "NoMode"], byte)


def getAmplMode(byte):
    return getFromList(["Any", "VCMode", "CCMode", "IDensityMode"], byte)


def getLeakHoldMode(byte):
    return getFromList(["Labs", "Lrel", "LabsLH", "LrelLH"], byte)


def getADCMode(byte):
    return getFromList(["AdcOff", "Analog", "Digitals", "Digital", "AdcVirtual"], byte)


def convertDataKind(byte):

    d = {}

    # LittleEndianBit = 0;
    # IsLeak          = 1;
    # IsVirtual       = 2;
    # IsImon          = 3;
    # IsVmon          = 4;
    # Clip            = 5;
    # (*
    #  -> meaning of bits:
    #     - LittleEndianBit => byte sequence
    #       "PowerPC Mac" = cleared
    #       "Windows and Intel Mac" = set
    #     - IsLeak
    #       set if trace is a leak trace
    #     - IsVirtual
    #       set if trace is a virtual trace
    #     - IsImon
    #       -> set if trace was from Imon ADC
    #       -> it flags a trace to be used to
    #          compute LockIn traces from
    #       -> limited to "main" traces, not "leaks"!
    #     - IsVmon
    #       -> set if trace was from Vmon ADC
    #     - Clip
    #       -> set if amplifier of trace was clipping
    # *)

    d["IsLittleEndian"] = bool(byte & (1 << 0))
    d["IsLeak"] = bool(byte & (1 << 1))
    d["IsVirtual"] = bool(byte & (1 << 2))
    d["IsImon"] = bool(byte & (1 << 3))
    d["IsVmon"] = bool(byte & (1 << 4))
    d["Clip"] = bool(byte & (1 << 5))

    return d


def convertStimToDacID(byte):

    d = {}

    # StimToDacID :
    #   Specifies how to convert the Segment
    #   "Voltage" to the actual voltage sent to the DAC
    #   -> meaning of bits:
    #      bit 0 (UseStimScale)    -> use StimScale
    #      bit 1 (UseRelative)     -> relative to Vmemb
    #      bit 2 (UseFileTemplate) -> use file template
    #      bit 3 (UseForLockIn)    -> use for LockIn computation
    #      bit 4 (UseForWavelength)
    #      bit 5 (UseScaling)
    #      bit 6 (UseForChirp)
    #      bit 7 (UseForImaging)
    #      bit 14 (UseReserved)
    #      bit 15 (UseReserved)

    d["UseStimScale"] = bool(byte & (1 << 0))
    d["UseRelative"] = bool(byte & (1 << 1))
    d["UseFileTemplate"] = bool(byte & (1 << 2))
    d["UseForLockIn"] = bool(byte & (1 << 3))
    d["UseForWavelength"] = bool(byte & (1 << 4))
    d["UseScaling"] = bool(byte & (1 << 5))
    d["UseForChirp"] = bool(byte & (1 << 6))
    d["UseForImaging"] = bool(byte & (1 << 7))

    return d


def getSquareKind(byte):
    return getFromList(["Common Frequency"], byte)


def getChirpKind(byte):
    return getFromList(["Linear", "Exponential", "Spectroscopic"], byte)


class Struct():
    """High-level wrapper around struct.Struct that makes it a bit easier to 
    unpack large, nested structures.
    
    * Unpacks to dictionary allowing fields to be retrieved by name
    * Optionally massages field data on read
    * Handles arrays and nested structures
    
    *fields* must be a list of tuples like (name, format) or (name, format, function)
    where *format* must be a simple struct format string like 'i', 'd', 
    '32s', or '4d'; or another Struct instance.
    
    *function* may be either a function that filters the data for that field
    or None to exclude the field altogether.
    
    If *size* is given, then an exception will be raised if the final struct size
    does not match the given size.

    
    Example::
        
        class MyStruct(Struct):
            field_info = [
                ('char_field', 'c'),                # single char 
                ('char_array', '8c'),               # list of 8 chars
                ('str_field',  '8s', cstr),         # C string of len 8
                ('sub_struct', MyOtherStruct),      # dict generated by s2.unpack 
                ('filler', '32s', None),            # ignored field
            ]
            size_check = 300
            
        fh = open(fname, 'rb')
        data = MyStruct(fh)
    
    """
    field_info = None
    size_check = None
    _fields_parsed = None
    
    
    def __init__(self, data, endian='<'):
        """Read the structure from *data* and return an ordered dictionary of 
        fields.
        
        *data* may be a string or file.
        *endian* may be '<' or '>'
        """
        field_info = self._field_info()
        if not isinstance(data, (str, bytes)):
            data = data.read(self._le_struct.size)
        if endian == '<':
            items = self._le_struct.unpack(data)
        elif endian == '>':
            items = self._be_struct.unpack(data)
        else:
            raise ValueError('Invalid endian: %s' % endian)
        
        fields = collections.OrderedDict()
        
        i = 0
        for name, fmt, func in field_info:
            # pull item(s) out of the list based on format string
            if len(fmt) == 1 or fmt[-1] == 's':
                item = items[i]
                i += 1
            else:
                n = int(fmt[:-1])
                item = items[i:i+n]
                i += n
            
            # try unpacking sub-structure
            if isinstance(func, tuple):
                substr, func = func
                item = substr(item, endian)
            
            # None here means the field should be omitted
            if func is None:
                continue
            # handle custom massaging function
            if func is not True:
                item = func(item)
            fields[name] = item
            setattr(self, name, item)
            
        self.fields = fields
        
    @classmethod
    def _field_info(cls):
        if cls._fields_parsed is not None:
            return cls._fields_parsed
        
        fmt = ''
        fields = []
        for items in cls.field_info:
            if len(items) == 3:
                name, ifmt, func = items
            else:
                name, ifmt = items
                func = True
                
            if isinstance(ifmt, type) and issubclass(ifmt, Struct):
                func = (ifmt, func) # instructs to unpack with sub-struct before calling function
                ifmt = '%ds' % ifmt.size()
            elif re.match(r'\d*[xcbB?hHiIlLqQfdspP]', ifmt) is None:
                raise TypeError('Unsupported format string "%s"' % ifmt)
            
            fields.append((name, ifmt, func))
            fmt += ifmt
        cls._le_struct = struct.Struct('<' + fmt)
        cls._be_struct = struct.Struct('>' + fmt)
        cls._fields_parsed = fields
        if cls.size_check is not None:
            assert cls._le_struct.size == cls.size_check, \
                "{} expected vs. {}".format(
                    cls.size_check, cls._le_struct.size)
        return fields

    @classmethod
    def size(cls):
        cls._field_info()
        return cls._le_struct.size
    
    @classmethod
    def array(cls, x):
        """Return a new StructArray class of length *x* and using this struct
        as the array item type.
        """
        return type(cls.__name__+'[%d]' % x, (StructArray,),
                    {'item_struct': cls, 'array_size': x})

    def __str__(self, indent=0):
        indent_str = '    '*indent
        r = indent_str + '%s(\n' % self.__class__.__name__
        if not hasattr(self, 'fields'):
            r = r[:-1] + '<initializing>)'
            return r
        for k, v in self.fields.items():
            if isinstance(v, Struct):
                r += indent_str + '    %s = %s\n' % \
                    (k, v.__str__(indent=indent+1).lstrip())
            else:
                r += indent_str + '    %s = %r\n' % (k, v)
        r += indent_str + ')'
        return r

    def get_fields(self):
        """Recursively convert struct fields+values to nested dictionaries.
        """
        fields = self.fields.copy()
        for k,v in fields.items():
            if isinstance(v, StructArray):
                fields[k] = [x.get_fields() for x in v.array]
            elif isinstance(v, Struct):
                fields[k] = v.get_fields()
        return fields

    
class StructArray(Struct):
    item_struct = None
    array_size = None
    
    def __init__(self, data, endian='<'):
        if not isinstance(data, (str, bytes)):
            data = data.read(self.size())
        items = []
        isize = self.item_struct.size()
        for i in range(self.array_size):
            d = data[:isize]
            data = data[isize:]
            items.append(self.item_struct(d, endian))
        self.array = items

    def __getitem__(self, i):
        return self.array[i]
        
    @classmethod
    def size(self):
        return self.item_struct.size() * self.array_size

    def __repr__(self, indent=0):
        r = '    '*indent + '%s(\n' % self.__class__.__name__
        for item in self.array:
            r += item.__repr__(indent=indent+1) + ',\n'
        r += '    '*indent + ')'
        return r


class BundleItem(Struct):
    field_info = [
        ('Start', 'i'),
        ('Length', 'i'),
        ('Extension', '8s', cstr),
    ]
    size_check = 16


class BundleHeader(Struct):
    field_info = [
        ('Signature', '8s', cstr),
        ('Version', '32s', cstr),
        ('Time', 'd', heka_time_to_date),
        ('Items', 'i'),
        ('IsLittleEndian', '?'),
        ('Reserved', '11s', None),
        ('BundleItems', BundleItem.array(12)),
    ]
    size_check = 256


class TreeNode(Struct):
    """Struct that also represents a node in a Pulse file tree.
    """
    def __init__(self, fh, pul, level=0):
        self.level = level
        self.children = []
        endian = pul.endian
        
        # The record structure in the file may differ from our expected structure
        # due to version differences, so we read the required number of bytes, and
        # then pad or truncate before unpacking the record. This will probably
        # result in corrupt data in some situations..
        realsize = pul.level_sizes[level]
        structsize = self.size()
        data = fh.read(realsize)
        diff = structsize - realsize
        if diff > 0:
            data = data + b'\0'*diff
        else:
            data = data[:structsize]
        
        # initialize struct data
        Struct.__init__(self, data, endian)
        
        # Next read the number of children
        nchild = struct.unpack(endian + 'i', fh.read(4))[0]
            
        level += 1
        if level >= len(pul.rectypes):
            return
        child_rectype = pul.rectypes[level]
        for i in range(nchild):
            self.children.append(child_rectype(fh, pul, level))

    def __getitem__(self, i):
        return self.children[i]
    
    def __len__(self):
        return len(self.children)
    
    def __iter__(self):
        return self.children.__iter__()
    
    def __repr__(self, indent=0):
        # Return a string describing this structure
        ind = '    '*indent
        srep = Struct.__repr__(self, indent)[:-1]  # exclude final parenthese
        srep += ind + '    children = %d,\n' % len(self)
        #srep += ind + 'children = [\n'
        #for ch in self:
            #srep += ch.__repr__(indent=indent+1) + ',\n'
        srep += ind + ')'
        return srep


class TraceRecord(TreeNode):
    field_info = [
        ('Mark', 'i'),
        ('Label', '32s', cstr),
        ('TraceID', 'i'),
        ('Data', 'i'),  
        ('DataPoints', 'i'),
        ('InternalSolution', 'i'),
        ('AverageCount', 'i'),
        ('LeakID', 'i'),
        ('LeakTraces', 'i'),
        ('DataKind', 'h', convertDataKind),
        ('UseXStart', '?'),
        ('TcKind', 'b'),
        ('RecordingMode', 'b', getRecordingMode),
        ('AmplIndex', 'b'),
        ('DataFormat', 'b', getDataFormat),
        ('DataAbscissa', 'b'),
        ('DataScaler', 'd'),
        ('TimeOffset', 'd'),
        ('ZeroData', 'd'),
        ('YUnit', '8s', cstr),
        ('XInterval', 'd'),
        ('XStart', 'd'),
        ('XUnit', '8s', cstr),
        ('YRange', 'd'),
        ('YOffset', 'd'),
        ('Bandwidth', 'd'),
        ('PipetteResistance', 'd'),
        ('CellPotential', 'd'),
        ('SealResistance', 'd'),
        ('CSlow', 'd'),
        ('GSeries', 'd'),
        ('RsValue', 'd'),
        ('GLeak', 'd'),
        ('MConductance', 'd'),
        ('LinkDAChannel', 'i'),
        ('ValidYrange', '?'),
        ('AdcMode', 'b', getADCMode),
        ('AdcChannel', 'h'),
        ('Ymin', 'd'),
        ('Ymax', 'd'),
        ('SourceChannel', 'i'),
        ('ExternalSolution', 'i'),
        ('CM', 'd'),
        ('GM', 'd'),
        ('Phase', 'd'),
        ('DataCRC', 'i'),
        ('CRC', 'I'),
        ('GS', 'd'),
        ('SelfChannel', 'i'),
        ('InterleaveSize', 'i'),
        ('InterleaveSkip', 'i'),
        ('ImageIndex', 'i'),
        ('TrMarkers', '10d'),
        ('SECM_X', 'd'),
        ('SECM_Y', 'd'),
        ('SECM_Z', 'd'),
        ('TrHolding', 'd'),
        ('TcEnumerator', 'i'),
        ('XTrace', 'i'),
        ('IntSolValue', 'd'),
        ('ExtSolValue', 'd'),
        ('IntSolName', '32s', cstr),
        ('ExtSolName', '32s', cstr),
        ('DataPedestal', 'd'),
    ]
    size_check = 512


class SweepRecord(TreeNode):
    field_info = [
        ('Mark', 'i'),
        ('Label', '32s', cstr),
        ('AuxDataFileOffset', 'i'),
        ('StimCount', 'i'),
        ('SweepCount', 'i'),
        ('Time', 'd'),
        ('Timer', 'd'),
        ('SwUserParams', '2d'),
        ('PipPressure', 'd'),
        ('RMSNoise', 'd'),
        ('Temperature', 'd'),
        ('OldIntSol', 'i'),
        ('OldExtSol', 'i'),
        ('DigitalIn', 'h'),
        ('SweepKind', 'h'),
        ('DigitalOut', 'h'),
        ('Filler1', 'h', None),
        ('Markers', '4d'),
        ('Filler2', 'i', None),
        ('CRC', 'I'),
        ('SwHolding', '16d'),
        ('SwUserParamEx', '8d'),
    ]
    size_check = 352


class UserParamDescrType(Struct):
    field_info = [
        ('Name', '32s', cstr),
        ('Unit', '8s', cstr),
    ]
    size_check = 40

    
class AmplifierState(Struct):
    field_info = [
        ('StateVersion', '8s', cstr),
        ('RealCurrentGain', 'd'),
        ('RealF2Bandwidth', 'd'),
        ('F2Frequency', 'd'),
        ('RsValue', 'd'),
        ('RsFraction', 'd'),
        ('GLeak', 'd'),
        ('CFastAmp1', 'd'),
        ('CFastAmp2', 'd'),
        ('CFastTau', 'd'),
        ('CSlow', 'd'),
        ('GSeries', 'd'),
        ('StimDacScale', 'd'),
        ('CCStimScale', 'd'),
        ('VHold', 'd'),
        ('LastVHold', 'd'),
        ('VpOffset', 'd'),
        ('VLiquidJunction', 'd'),
        ('CCIHold', 'd'),
        ('CSlowStimVolts', 'd'),
        ('CCTrackVHold', 'd'),
        ('TimeoutLength', 'd'),
        ('SearchDelay', 'd'),
        ('MConductance', 'd'),
        ('MCapacitance', 'd'),
        ('SerialNumber', '8s', cstr),
        ('E9Boards', 'h'),
        ('CSlowCycles', 'h'),
        ('IMonAdc', 'h'),
        ('VMonAdc', 'h'),
        ('MuxAdc', 'h'),
        ('TstDac', 'h'),
        ('StimDac', 'h'),
        ('StimDacOffset', 'h'),
        ('MaxDigitalBit', 'h'),
        ('HasCFastHigh', 'b'),
        ('CFastHigh', 'b'),
        ('HasBathSense', 'b'),
        ('BathSense', 'b'),
        ('HasF2Bypass', 'b'),
        ('sF2Mode', 'b'),
        ('AmplKind', 'b', getAmplifierType),
        ('IsEpc9N', 'b'),
        ('ADBoard', 'b', getADBoard),
        ('BoardVersion', 'b'),
        ('ActiveE9Board', 'b'),
        ('Mode', 'b', getClampMode),
        ('Range', 'b'),
        ('F2Response', 'b'),
        ('RsOn', 'b'),
        ('CSlowRange', 'b'),
        ('CCRange', 'b'),
        ('CCGain', 'b'),
        ('CSlowToTstDac', 'b'),
        ('StimPath', 'b'),
        ('CCTrackTau', 'b'),
        ('WasClipping', 'b'),
        ('RepetitiveCSlow', 'b'),
        ('LastCSlowRange', 'b'),
        ('Old1', 'b', None),
        ('CanCCFast', 'b'),
        ('CanLowCCRange', 'b'),
        ('CanHighCCRange', 'b'),
        ('CanCCTracking', 'b'),
        ('HasVmonPath', 'b'),
        ('HasNewCCMode', 'b'),
        ('Selector', 'c', cbyte),
        ('HoldInverted', 'b'),
        ('AutoCFast', '?'),
        ('AutoCSlow', '?'),
        ('HasVmonX100', 'b'),
        ('TestDacOn', 'b'),
        ('QMuxAdcOn', 'b'),
        ('RealImon1Bandwidth', 'd'),
        ('StimScale', 'd'),
        ('Gain', 'b', getAmplifierGain),
        ('Filter1', 'b'),
        ('StimFilterOn', 'b'),
        ('RsSlow', 'b'),
        ('Old2', 'b', None),
        ('CCCFastOn', '?'),
        ('CCFastSpeed', 'b'),
        ('F2Source', 'b'),
        ('TestRange', 'b'),
        ('TestDacPath', 'b'),
        ('MuxChannel', 'b'),
        ('MuxGain64', 'b'),
        ('VmonX100', 'b'),
        ('IsQuadro', 'b'),
        ('F1Mode', 'b'),
        ('Old3', 'b', None),
        ('StimFilterHz', 'd'),
        ('RsTau', 'd'),
        ('DacToAdcDelay', 'd'),
        ('InputFilterTau', 'd'),
        ('OutputFilterTau', 'd'),
        ('VmonFactor', 'd', None),
        ('CalibDate', '16s', cstr),
        ('VmonOffset', 'd'),
        ('EEPROMKind', 'b'),
        ('VrefX2', 'b'),
        ('HasVrefX2AndF2Vmon', 'b'),
        ('Spare1', 'b', None),
        ('Spare2', 'b', None),
        ('Spare3', 'b', None),
        ('Spare4', 'b', None),
        ('Spare5', 'b', None),
        ('CCStimDacScale', 'd'),
        ('VmonFiltBandwidth', 'd'),
        ('VmonFiltFrequency', 'd'),
    ]
    size_check = 400

    
class LockInParams(Struct):
    field_info = [
        ('ExtCalPhase', 'd'),
        ('ExtCalAtten', 'd'),
        ('PLPhase', 'd'),
        ('PLPhaseY1', 'd'),
        ('PLPhaseY2', 'd'),
        ('UsedPhaseShift', 'd'),
        ('UsedAttenuation', 'd'),
        ('Spares2', '8s', None),
        ('ExtCalValid', '?'),
        ('PLPhaseValid', '?'),
        ('LockInMode', 'b'),
        ('CalMode', 'b'),
        ('Spares', '28s', None),
    ]
    size_check = 96


class SeriesRecord(TreeNode):
    field_info = [
        ('Mark', 'i'),
        ('Label', '32s', cstr),
        ('Comment', '80s', cstr),
        ('SeriesCount', 'i'),
        ('NumberSweeps', 'i'),
        ('AmplStateFlag', 'i'),
        ('AmplStateRef', 'i'),
        ('MethodTag', 'i'),
        ('Time', 'd'),
        ('PageWidth', 'd'),
        ('UserDescr1', UserParamDescrType.array(2)),
        ('Filler1', UserParamDescrType.array(2), None),
        ('MethodName', '32s', cstr),
        ('PhotoParams1', '4d'),
        ('LockInParams', LockInParams),
        ('AmplifierState', AmplifierState),
        ('Username', '80s', cstr),
        ('PhotoParams2', UserParamDescrType.array(4)),
        ('Filler2', 'i', None),
        ('CRC', 'I'),
        ('UserParams2', '4d'),
        ('UserParamDescr2', UserParamDescrType.array(4)),
        ('ScanParams', '12d'),
        ('UserDescr2', UserParamDescrType.array(8)),
    ]
    size_check = 1728


class GroupRecord(TreeNode):
    field_info = [
        ('Mark', 'i'),
        ('Label', '32s', cstr),
        ('Text', '80s', cstr),
        ('ExperimentNumber', 'i'),
        ('GroupCount', 'i'),
        ('CRC', 'I'),
        ('MatrixWidth', 'd'),
        ('MatrixHeight', 'd'),
    ]
    size_check = 144


class AmplStateRecord(TreeNode):
    field_info = [
        ('Mark', 'i'),
        ('StateCount', 'i'),
        ('StateVersion', 'b'),
        ('Filler1', 'c', None),
        ('Filler2', 'c', None),
        ('Filler3', 'c', None),
        ('Filler4', 'i', None),
        ('LockInParams', LockInParams),
        ('AmplifierState', AmplifierState),
        ('IntSol', 'i'),
        ('ExtSol', 'i'),
        ('Filler5', '36c', None),
        ('CRC', 'I'),
    ]
    size_check = 560    
    
    
class AmpSeriesRecord(TreeNode):
    field_info = [
        ('Mark', 'i'),
        ('StateCount', 'i'),
        ('Filler1', 'i', None),
        ('CRC', 'I'),
    ]
    size_check = 16


class StimulationRecord(TreeNode):
    field_info = [
        ('Mark', 'i'),
        ('EntryName', '32s', cstr),
        ('FileName', '32s', cstr),
        ('AnalName', '32s', cstr),
        ('DataStartSegment', 'i'),
        ('DataStartTime', 'd'),
        ('SampleInterval', 'd'),
        ('SweepInterval', 'd'),
        ('LeakDelay', 'd'),
        ('FilterFactor', 'd'),
        ('NumberSweeps', 'i'),
        ('NumberLeaks', 'i'),
        ('NumberAverages', 'i'),
        ('ActualAdcChannels', 'i'),
        ('ActualDacChannels', 'i'),
        ('ExtTrigger', 'b'),
        ('NoStartWait', '?'),
        ('UseScanRates', '?'),
        ('NoContAq', '?'),
        ('HasLockIn', '?'),
        ('OldStartMacKind', 'b'),
        ('OldEndMacKind', '?'),
        ('AutoRange', 'b'),
        ('BreakNext', '?'),
        ('IsExpanded', '?'),
        ('LeakCompMode', '?'),
        ('HasChirp', '?'),
        ('OldStartMacro', '32s', cstr),
        ('OldEndMacro', '32s', cstr),
        ('IsGapFree', '?'),
        ('HandledExternally', '?'),
        ('Filler1', '?', None),
        ('Filler2', '?', None),
        ('CRC', 'I'),
    ]
    size_check = 248
    

class ChannelRecord(TreeNode):
    field_info = [
        ('Mark', 'i'),
        ('LinkedChannel', 'i'),
        ('CompressionFactor', 'i'),
        ('YUnit', '8s', cstr),
        ('AdcChannel', 'h'),
        ('AdcMode', 'b', getADCMode),
        ('DoWrite', '?'),
        ('LeakStore', 'b'),
        ('AmplMode', 'b', getAmplMode),
        ('OwnSegTime', '?'),
        ('SetLastSegVmemb', '?'),
        ('DacChannel', 'h'),
        ('DacMode', 'b'),
        ('HasLockInSquare', 'b'),
        ('RelevantXSegment', 'i'),
        ('RelevantYSegment', 'i'),
        ('DacUnit', '8s', cstr),
        ('Holding', 'd'),
        ('LeakHolding', 'd'),
        ('LeakSize', 'd'),
        ('LeakHoldMode', 'b', getLeakHoldMode),
        ('LeakAlternate', '?'),
        ('AltLeakAveraging', '?'),
        ('LeakPulseOn', '?'),
        ('StimToDacID', 'h', convertStimToDacID),
        ('CompressionMode', 'h'),
        ('CompressionSkip', 'i'),
        ('DacBit', 'h'),
        ('HasLockInSine', '?'),
        ('BreakMode', 'b'),
        ('ZeroSeg', 'i'),
        ('StimSweep', 'i'),
        ('Sine_Cycle', 'd'),
        ('Sine_Amplitude', 'd'),
        ('LockIn_VReversal', 'd'),
        ('Chirp_StartFreq', 'd'),
        ('Chirp_EndFreq', 'd'),
        ('Chirp_MinPoints', 'd'),
        ('Square_NegAmpl', 'd'),
        ('Square_DurFactor', 'd'),
        ('LockIn_Skip', 'i'),
        ('Photo_MaxCycles', 'i'),
        ('Photo_SegmentNo', 'i'),
        ('LockIn_AvgCycles', 'i'),
        ('Imaging_RoiNo', 'i'),
        ('Chirp_Skip', 'i'),
        ('Chirp_Amplitude', 'd'),
        ('Photo_Adapt', 'b'),
        ('Sine_Kind', 'b'),
        ('Chirp_PreChirp', 'b'),
        ('Sine_Source', 'b'),
        ('Square_NegSource', 'b'),
        ('Square_PosSource', 'b'),
        ('Chirp_Kind', 'b', getChirpKind),
        ('Chirp_Source', 'b'),
        ('DacOffset', 'd'),
        ('AdcOffset', 'd'),
        ('TraceMathFormat', 'b'),
        ('HasChirp', '?'),
        ('Square_Kind', 'b', getSquareKind),
        ('Filler1', '5c', None),
        ('Square_BaseIncr', 'd'),
        ('Square_Cycle', 'd'),
        ('Square_PosAmpl', 'd'),
        ('CompressionOffset', 'i'),
        ('PhotoMode', 'i'),
        ('BreakLevel', 'd'),
        ('TraceMath', '128s', cstr),
        ('Filler2', 'i', None),
        ('CRC', 'I'),
    ]
    size_check = 400


class StimSegmentRecord(TreeNode):
    field_info = [
        ('Mark', 'i'),
        ('Class', 'b', getSegmentClass),
        ('StoreKind', 'b', getStoreType),
        ('VoltageIncMode', 'b', getIncrementMode),
        ('DurationIncMode', 'b', getIncrementMode),
        ('Voltage', 'd'),
        ('VoltageSource', 'i', getSourceType),
        ('DeltaVFactor', 'd'),
        ('DeltaVIncrement', 'd'),
        ('Duration', 'd'),
        ('DurationSource', 'i', getSourceType),
        ('DeltaTFactor', 'd'),
        ('DeltaTIncrement', 'd'),
        ('Filler1', 'i', None),
        ('CRC', 'I'),
        ('ScanRate', 'd'),
    ]
    size_check = 80
    

class Pulsed(TreeNode):
    field_info = [
        ('Version', 'i'),
        ('Mark', 'i'),
        ('VersionName', '32s', cstr),
        ('AuxFileName', '80s', cstr),
        ('RootText', '400s', cstr),
        ('StartTime', 'd'),
        ('MaxSamples', 'i'),
        ('CRC', 'I'),
        ('Features', 'h'),
        ('Filler1', 'h', None),
        ('Filler2', 'i', None),
        ('TcEnumerator', '32h'),
        ('TcKind', '32b'),
    ]
    size_check = 640
    
    rectypes = [
        None,
        GroupRecord,
        SeriesRecord,
        SweepRecord,
        TraceRecord
    ]
    
    def __init__(self, bundle, offset=0, size=None):
        fh = bundle.fh  # Use bundle.fh
        fh.seek(offset)
        
        # read .pul header
        magic = fh.read(4) 
        if magic == b'eerT':
            self.endian = '<'
        elif magic == b'Tree':
            self.endian = '>'
        else:
            raise RuntimeError('Bad file magic: %s' % magic)
        
        levels = struct.unpack(self.endian + 'i', fh.read(4))[0]

        # read size of each level (one int per level)
        self.level_sizes = []
        for i in range(levels):
            size = struct.unpack(self.endian + 'i', fh.read(4))[0]
            self.level_sizes.append(size)
            
        TreeNode.__init__(self, fh, self)
        # Do not close fh here


class Data(object):
    def __init__(self, bundle, offset=0, size=None):
        self.bundle = bundle
        self.offset = offset
        
    def __getitem__(self, *args):
        index = args[0]
        assert len(index) == 4
        pul = self.bundle.pul
        trace = pul[index[0]][index[1]][index[2]][index[3]]
        fh = self.bundle.fh  # Use self.bundle.fh
        fh.seek(trace.Data)
        dtype = np.dtype(convertDataFormatToNP(trace.DataFormat))
        if not trace.DataKind['IsLittleEndian']: #  for big endian data, we need to swap
            dtype = dtype.newbyteorder('>')
        data = np.fromfile(fh, count=trace.DataPoints, dtype=dtype)
        # Do not close fh here

        return (data * trace.DataScaler).astype(np.float64)


class Amplifier(TreeNode):
    field_info = [
        ('Version', 'i'),
        ('Mark', 'i'),
        ('VersionName', '32s', cstr),
        ('AmplifierName', '32s', cstr),
        ('Amplifier', 'b'),
        ('ADBoard', 'b'),
        ('Creator', 'b'),
        ('Filler1', 'c', None),
        ('CRC', 'I'),
    ]
    size_check = 80
    
    rectypes = [
        None,
        AmpSeriesRecord,
        AmplStateRecord
    ]
    
    def __init__(self, bundle, offset=0, size=None):
        fh = bundle.fh  # Use bundle.fh
        fh.seek(offset)
        
        # read .pul header
        magic = fh.read(4) 
        if magic == b'eerT':
            self.endian = '<'
        elif magic == b'Tree':
            self.endian = '>'
        else:
            raise RuntimeError('Bad file magic: %s' % magic)
        
        levels = struct.unpack(self.endian + 'i', fh.read(4))[0]

        # read size of each level (one int per level)
        self.level_sizes = []
        for i in range(levels):
            size = struct.unpack(self.endian + 'i', fh.read(4))[0]
            self.level_sizes.append(size)
            
        TreeNode.__init__(self, fh, self)
        # Do not close fh here


class Stimulus(TreeNode):
    field_info = [
        ('Version', 'i'),
        ('Mark', 'i'),
        ('VersionName', '32s', cstr),
        ('MaxSamples', 'i'),
        ('Filler1', 'i', None),
        ('Params', '10d'),
        ('ParamText', '320c', None),
        ('Reserved', '128s', cstr),
        ('Filler2', 'i', None),
        ('Reserved2', '560s', None),
        ('CRC', 'I'),
    ]
    size_check = 1144
    
    rectypes = [
        None,
        StimulationRecord,
        ChannelRecord,
        StimSegmentRecord
    ]
    
    def __init__(self, bundle, offset=0, size=None):
        fh = bundle.fh  # Use bundle.fh
        fh.seek(offset)
        
        # read .pul header
        magic = fh.read(4) 
        if magic == b'eerT':
            self.endian = '<'
        elif magic == b'Tree':
            self.endian = '>'
        else:
            raise RuntimeError('Bad file magic: %s' % magic)
        
        levels = struct.unpack(self.endian + 'i', fh.read(4))[0]

        # read size of each level (one int per level)
        self.level_sizes = []
        for i in range(levels):
            size = struct.unpack(self.endian + 'i', fh.read(4))[0]
            self.level_sizes.append(size)
            
        TreeNode.__init__(self, fh, self)
        # Do not close fh here


class Bundle(object):
    """
    Represent a PATCHMASTER tree file in memory
    """

    item_classes = {
        '.pul': Pulsed,
        '.dat': Data,
        '.amp': Amplifier,
        '.pgf': Stimulus,
    }
    
    def __init__(self, file_name):
        self.file_name = file_name
        self.fh = open(self.file_name, 'rb')

        if self.fh.read(4) != b'DAT2':
            raise ValueError(f"No support for other files than 'DAT2' format")

        self.fh.seek(0)

        # Read header assuming little endian
        endian = '<'
        self.header = BundleHeader(self.fh, endian)
        # If the header is bad, re-read using big endian
        if not self.header.IsLittleEndian:
            endian = '>'
            self.fh.seek(0)
            self.header = BundleHeader(self.fh, endian)

        # catalog extensions of bundled items
        self.catalog = {}
        for item in self.header.BundleItems:
            item.instance = None
            ext = item.Extension
            self.catalog[ext] = item

    def close(self):
        if hasattr(self, 'fh') and self.fh:
            self.fh.close()

    @property
    def pul(self):
        """The Pulsed object from this bundle.
        """
        return self._get_item_instance('.pul')
    
    @property
    def data(self):
        """The Data object from this bundle.
        """
        return self._get_item_instance('.dat')

    @property
    def amp(self):
        """The Amplifier object from this bundle.
        """
        return self._get_item_instance('.amp')

    @property
    def pgf(self):
        """The PGF object from this bundle.
        """
        return self._get_item_instance('.pgf')

    def _get_item_instance(self, ext):
        if ext not in self.catalog:
            return None
        item = self.catalog[ext]
        if item.instance is None:
            cls = self.item_classes[ext]
            # Pass self (the bundle instance) instead of file_name
            item.instance = cls(self, item.Start, item.Length)
        return item.instance
        
    def __repr__(self):
        return "Bundle(%r)" % list(self.catalog.keys())
