#pragma once

#include <algorithm>
#include <iostream>
#include <memory>
#include <stdint.h>
#include <string.h>
#include <vector>

class CensusMat {
private:
  std::shared_ptr<uint32_t> data;

public:
  struct Offset {
    int x, y;
    Offset() : x(0), y(0) {}
    Offset(int _x, int _y) : x(_x), y(_y) {}
  };
  Offset offset;
  int rows, cols, step;

public:
  CensusMat(int _rows, int _cols, uint32_t _value)
      : data(new uint32_t[_rows * _cols]), offset(0, 0), rows(_rows),
        cols(_cols), step(_cols) {
    memset(data.get(), _value, _rows * _cols * sizeof(uint32_t));
  }

  CensusMat(CensusMat &_parent, int _x, int _y, int _width, int _height)
      : data(_parent.data), offset(_x, _y), rows(_height), cols(_width),
        step(_parent.step) {}

  void set(uint32_t _value) {
    memset(data.get(), _value, rows * cols * sizeof(uint32_t));
  }

  uint32_t *unsafeAt(int _x, int _y) {
    return (data.get() + (offset.y + _y) * step + _x + offset.x);
  }

  // Determines if the given location (wrt the parent) is within this CensusMat)
  bool isWithin(int _parentX, int _parentY) const {
    return (offset.x <= _parentX && offset.y <= _parentY &&
            (offset.x + cols) > _parentX && (offset.y + rows) > _parentY);
  }
};
