## Pipelining, in code

```py
wup,todevice,declare_stream,host,ipu0,ipu1,send,recv,A,B,C,D,E,F = foo()

# A -> B -> C --> D-\
#  \    \           |
#   \    ------\    |
#    \         V    |
#     \-> F <- E <--/


# A -> B -> C -> D -> E -> F 
#  \    \______>_____/   /
#   \__________>________/

# Basic model, little parallelism due to data dep
def f(WA, WB, WC, x, l):
  a = A(WA, x)
  b = B(WB, a)
  c = C(WC, b)
  d = D(c, l)
  WC = wup(d, WC)
  e = E(WB, d, b)
  WB = wup(e, WB)
  f = F(WA, e, a)
  WA = wup(f, WA)

# But put it on multi machines anyway.
# Rule: all arguments of a fcall must be on one device, 
# then the fcall happens on that device.
# Variable names are suffixed with device ID
def f(WA0, WB0, WC1, x0, l1):
  a0  = A(WA0, x0)            # a : ipu0
  b0  = B(WB0, a0)            # b : ipu0
  b1  = send(b0, ipu1)        # b1: ipu 1
  c1  = C(WC1, b1)            # c : ipu 1
  d1  = D(c1, l1)             # d : ipu 1
  WC1 = wup(d1, WC1)          # WC: ipu 1
  d0  = send(d1, ipu0)        # d0: ipu0
  e0  = E(WB0, d0, b0)        # e : ipu0
  WB0 = wup(e0, WB0)          # WB: ipu0
  f0  = F(WA0, e0, a0)        # f : ipu0
  WA0 = wup(f0, WA0)          # WA: ipu0

####################################################
# Make the MPMD program.
# Two on-device functions: (f_ipu0, f_ipu1)
# And some defined streams
# declare_stream('x', host->ipu0, size(x))
# declare_stream('l', host->ipu1, size(l))
# declare_stream('b01', ipu0->ipu1, size(b))
# declare_stream('d10', ipu1->ipu0, size(d))
def f_ipu0():
  x0  = recv('x')
  a0  = A(WA0, x0)            # a : ipu0
  b0  = B(WB0, a0)            # b : ipu0
  send(b0, 'b01')
  # b1  = todevice(b0, ipu1)    # b1: ipu 1
  # c1  = C(WC1, b1)            # c : ipu 1
  # d1  = D(c1, l1)             # d : ipu 1
  # WC1 = wup(d1, WC1)          # WC: ipu 1
  d0  = recv('d10')             # d0: ipu0
  e0  = E(WB0, d0, b0)        # e : ipu0
  WB0 = wup(e0, WB0)          # WB: ipu0
  f0  = F(WA0, e0, a0)        # f : ipu0
  WA0 = wup(f0, WA0)          # WA: ipu0

def f_ipu1():
  l1  = recv('l')
  # a0  = A(WA0, x0)            # a : ipu0
  # b0  = B(WB0, a0)            # b : ipu0
  # send(b0, ipu1)
  b1  = recv('b01')            # b1: ipu 1
  c1  = C(WC1, b1)            # c : ipu 1
  d1  = D(c1, l1)             # d : ipu 1
  WC1 = wup(d1, WC1)          # WC: ipu 1
  send(d1, 'd10')
  # d0  = recv('d1')              # d0: ipu0
  # e0  = E(WB0, d0, b0)        # e : ipu0
  # WB0 = wup(e0, WB0)          # WB: ipu0
  # f0  = F(WA0, e0, a0)        # f : ipu0
  # WA0 = wup(f0, WA0)          # WA: ipu0



####################################################
# The MPMD program.
# Two on-device loops: (f_ipu0, f_ipu1)
# And some defined streams
# declare_stream('xl', host->ipu[01], size(x + l))
# declare_stream('b01', ipu0->ipu1, size(b))
# declare_stream('d10', ipu1->ipu0, size(d))
def f_ipu0():
  for i in range(0,MAX):
    x0,_  = recv('xl')        # EXCH
    a0  = A(WA0, x0)
    b0  = B(WB0, a0)
    send(b0, 'b01')           # EXCH
    d0  = recv('d10')         #
    if i > 0:
      e0  = E(WB0, d0, b0)
      WB0 = wup(e0, WB0)
      f0  = F(WA0, e0, a0)
      WA0 = wup(f0, WA0)
    else:
      ... # igmnore dummy d0

def f_ipu1():
  for i in range(0,MAX):
    _,l1_next  = recv('xl')   # EXCH
    if i > 0:
      # Use b1,l1 from previous iter
      c1  = C(WC1, b1)
      d1  = D(c1, l1)
      WC1 = wup(d1, WC1)
    else:
      d1 = ... # dummy
    b1  = recv('b01')         # EXCH
    send(d1, 'd10')           #
    l1 = l1_next
```
