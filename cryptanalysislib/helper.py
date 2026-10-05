#!/usr/bin/env python3
"""
"""

from typing import Dict
import pathlib
import re 
import argparse


def dict2str(d: Dict) -> str:
    """
    NOTE: only writes `int`. Floats or other types are skipped 
    :param d: the dictionary which is going to be written to a string
    :return:
        #ifndef INCLUDE_PARAMS 
        #define INCLUDE_PARAMS 
        #define PARAM_${key} ${value}; // for all key/values in dict
        #endif
    """
    a = "\n".join(f"#define PARAM_{k} {v}" for k,v in d.items() if isinstance(v, int))
    ret = "#ifndef INCLUDE_PARAMS\n#define INCLUDE_PARAMS\n\n" 
    ret += a
    ret += "\n\n#endif"
    return ret


def dict2include(file: pathlib.Path | str, d: Dict) -> bool:
    """
    Translates the given dictionary 'd', Translates it via `dict2str` and 
    writes this string into `file`
    :param file: str or path to write to
    :param d: dictionary to wrtie
    :return : true if success
    """
    a = dict2str(d)
    if isinstance(file, str):
        file = pathlib.Path(file)

    with file.open("w", encoding ="utf-8") as f:
        f.write(a)
        return True

    return False


class Range(argparse.Action):
    """ just a wrapper around `range()` function with a name. While
        extends the normal `argparse` package with the possibility to 
        pass a upper/lower bound to each parameter
    """
    def __init__(self, name: str="", 
                 start: int=0, 
                 end: int = -1,
                 step: int = 0,
                 *args, **kwargs) -> None:
        """
        :param name:
        :param start:
        :param end: -1 is the not passed symbol
        :param step:
        """
        assert (end != start)

        if "option_strings" in kwargs:
            self.name = kwargs["option_strings"][0].replace("-", "")
        else:
            self.name = name
            kwargs["option_strings"] = list()
            kwargs["option_strings"].append("-" + name)

        if "dest" not in kwargs:
            kwargs["dest"] = name

        super(Range, self).__init__(*args, **kwargs)
        self.start = start
        self.current = start
        self.next = self.current
        
        if end != -1:
            if start < end:
                self.end = start + 1 if end == -1 else end
                self.step = 1 if step == 0 else step
            else:
                self.end = start - 1 if end == -1 else end
                self.step = -1 if step == 0 else step
                # make sure that setp is negative
                if self.step > 0:
                    self.step = -self.step
        else:
            self.end = start + 1
            self.step = 1
        
    def size(self) -> int:
        """ returns the number of value the Range can enumerate at most """
        return abs((abs(abs(self.end) - self.start + abs(self.step) - 1)) // self.step)
    
    def reset(self):
        """ resets the current state of the range"""
        self.current = self.start
        self.next = self.start
    
    def __iter__(self):
        return self
    
    def __next__(self) -> int:
        self.current = self.next
        self.next = self.current + self.step
        if self.step > 0:
            if self.current >= self.end:
                raise StopIteration
        else: 
            if self.current <= self.end:
                raise StopIteration
        return self.current

    def parse_single_number(self, value: str):
        try:
            matches = re.findall("[0-9]+", value)
            if len(matches) != 1:
                return False, value

            t = int(matches[0])
            return True, Range(name=self.name,start=t)
        except:
            return False, {}

    def parse_double_number(self, value: str):
        try:
            matches = re.findall("[0-9]*,[0-9]*", value)
            if len(matches) != 1:
                return False, value

            splits = matches[0].split(",")
            assert len(splits) == 2
            t = [int(s) for s in splits]

            return True, Range(self.name,start=t[0],end=t[1])
        except:
            return False, []

    def parse_triple_number(self, value: str):
        try:
            matches = re.findall("[0-9]*,[0-9]*,[0-9]*", value)
            if len(matches) != 1:
                return False, value

            splits = matches[0].split(",")
            assert len(splits) == 3
            t = [int(s) for s in splits]
            return True, Range(self.name,start=t[0],end=t[1],step=t[2])
        except:
            return False, []

    def __call__(self, parser, namespace, value: str, option_string=None):
        self.step = 1
        value = str(value)
        ret, parsed_value = self.parse_single_number(value)

        if getattr(namespace, self.dest) is None:
            setattr(namespace, self.dest, [])
        elif type(getattr(namespace, self.dest)) is str:
            setattr(namespace, self.dest, [])

        tmp = getattr(namespace, self.dest)
        # this is stupid. But `argparse` does not apply a custom action if no
        # arguments is applied
        for bla in tmp:
            if bla.name == self.name:
                return

        if not ret:
            ret, parsed_value = self.parse_double_number(value)
            if not ret:
                ret, parsed_value = self.parse_triple_number(value)
                if not ret:
                    msg = "Invalid format: " + str(value)
                    raise argparse.ArgumentError(self, msg)
                else:
                    tmp.append(parsed_value)
            else:
                tmp.append(parsed_value)
        else:
            tmp.append(parsed_value)

        setattr(namespace, self.dest, tmp)
        return
