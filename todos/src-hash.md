# TODOs: src-hash

14 entries. Format: `- [ ] file:line [MARKER] text`.

## src/hash/adler32.h

- [ ] `src/hash/adler32.h:11` **TODO** add dispatch function like in `src/search/binary.h`
- [ ] `src/hash/adler32.h:12` **TODO** move the internal horizontally addition functions to the simd interface
- [ ] `src/hash/adler32.h:132` **TODO** make this a function argument? Or move it anywhere useful

## src/hash/cityhash.h

- [ ] `src/hash/cityhash.h:28` **TODO** comments — TODO move fetch64/32 and rotate32  into internal namespace
- [ ] `src/hash/cityhash.h:29` **TODO** move fetch64/32 and rotate32  into internal namespace
- [ ] `src/hash/cityhash.h:41` **TODO** use own rotate implementation, which is optimized

## src/hash/crc.h

- [ ] `src/hash/crc.h:8` **TODO** add dispatch function like in `src/search/binary.h`
- [ ] `src/hash/crc.h:9` **TODO** add namespace and stuff

## src/hash/fnv1a.h

- [ ] `src/hash/fnv1a.h:12` **TODO** SIMD version
- [ ] `src/hash/fnv1a.h:13` **TODO** add namespace

## src/hash/simple.h

- [ ] `src/hash/simple.h:15` **TODO** add docs — TODO add namespace
- [ ] `src/hash/simple.h:16` **TODO** add namespace
- [ ] `src/hash/simple.h:17` **TODO** add [[nodiscard]]

## src/hash/standard.h

- [ ] `src/hash/standard.h:13` **TODO** add concept for T1
