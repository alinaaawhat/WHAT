# This is disassembled bytecode, not original source code
# pyc file: /home/SHIH0020/robustlearn/DI2SDiff/data_load/data_util/__pycache__/raw_aug_loader.cpython-39.pyc

Name:              <module>
Filename:          /home/zhangjunru/robustlearn/ddlearn/data_util/raw_aug_loader.py
Argument count:    0
Positional-only arguments: 0
Kw-only arguments: 0
Number of locals:  0
Stack size:        18
Flags:             NOFREE
Constants:
   0: 0
   1: None
   2: ('MinMaxScaler',)
   3: ('args_parse',)
   4: ('raw_to_aug',)
   5: <code object load_raw_aug_data at 0x7ff4837feef0, file "/home/zhangjunru/robustlearn/ddlearn/data_util/raw_aug_loader.py", line 15>
   6: 'load_raw_aug_data'
   7: <code object data_scaler at 0x7ff4837ff520, file "/home/zhangjunru/robustlearn/ddlearn/data_util/raw_aug_loader.py", line 104>
   8: 'data_scaler'
   9: <code object reshape_data at 0x7ff4837ff730, file "/home/zhangjunru/robustlearn/ddlearn/data_util/raw_aug_loader.py", line 127>
  10: 'reshape_data'
  11: <code object pick_data at 0x7ff4837ff7e0, file "/home/zhangjunru/robustlearn/ddlearn/data_util/raw_aug_loader.py", line 147>
  12: 'pick_data'
  13: <code object set_param at 0x7ff4837ff890, file "/home/zhangjunru/robustlearn/ddlearn/data_util/raw_aug_loader.py", line 179>
  14: 'set_param'
  15: '__main__'
  16: '/media/newdisk/zhangjunru/DG_Dateset/'
  17: ('uschad',)
  18: ('norm',)
  19: (0.2, 0.4, 0.6, 0.8, 1.0)
  20: 2
  21: 3
  22: 1
  23: 7
  24: ('save_path', 'dataset', 'aug_num', 'remain_data_rate')
  25: '/'
  26: '_crosssubject_rawaug_rate'
  27: '_t'
  28: '_seed'
  29: '_scaler'
  30: '.pkl'
  31: ('raw_trs', 'aug_trs', 'raw_vas', 'aug_vas', 'raw_trt', 'aug_trt', 'raw_vat', 'aug_vat', 'raw_tet', 'aug_tet')
  32: 'wb'
Names:
   0: sys
   1: os
   2: path
   3: append
   4: dirname
   5: sklearn.preprocessing
   6: MinMaxScaler
   7: pickle
   8: numpy
   9: np
  10: utils
  11: main
  12: args_parse
  13: aug_preprocess
  14: raw_to_aug
  15: load_raw_aug_data
  16: data_scaler
  17: reshape_data
  18: pick_data
  19: set_param
  20: __name__
  21: args
  22: root_path
  23: dataset
  24: n_domain
  25: scaler_method
  26: remain_data_rate
  27: range
  28: seed
  29: set_random_seed
  30: raw_data
  31: aug_data
  32: target
  33: save_path
  34: print
  35: raw_trs
  36: aug_trs
  37: raw_vas
  38: aug_vas
  39: raw_trt
  40: aug_trt
  41: raw_vat
  42: aug_vat
  43: raw_tet
  44: aug_tet
  45: raw_and_aug
  46: open
  47: f
  48: dump

  5           0 LOAD_CONST               0 (0)
              2 LOAD_CONST               1 (None)
              4 IMPORT_NAME              0 (sys)
              6 STORE_NAME               0 (sys)

  6           8 LOAD_CONST               0 (0)
             10 LOAD_CONST               1 (None)
             12 IMPORT_NAME              1 (os)
             14 STORE_NAME               1 (os)

  7          16 LOAD_NAME                0 (sys)
             18 LOAD_ATTR                2 (path)
             20 LOAD_METHOD              3 (append)
             22 LOAD_NAME                1 (os)
             24 LOAD_ATTR                2 (path)
             26 LOAD_METHOD              4 (dirname)
             28 LOAD_NAME                0 (sys)
             30 LOAD_ATTR                2 (path)
             32 LOAD_CONST               0 (0)
             34 BINARY_SUBSCR
             36 CALL_METHOD              1
             38 CALL_METHOD              1
             40 POP_TOP

  8          42 LOAD_CONST               0 (0)
             44 LOAD_CONST               2 (('MinMaxScaler',))
             46 IMPORT_NAME              5 (sklearn.preprocessing)
             48 IMPORT_FROM              6 (MinMaxScaler)
             50 STORE_NAME               6 (MinMaxScaler)
             52 POP_TOP

  9          54 LOAD_CONST               0 (0)
             56 LOAD_CONST               1 (None)
             58 IMPORT_NAME              7 (pickle)
             60 STORE_NAME               7 (pickle)

 10          62 LOAD_CONST               0 (0)
             64 LOAD_CONST               1 (None)
             66 IMPORT_NAME              8 (numpy)
             68 STORE_NAME               9 (np)

 11          70 LOAD_CONST               0 (0)
             72 LOAD_CONST               1 (None)
             74 IMPORT_NAME             10 (utils)
             76 STORE_NAME              10 (utils)

 12          78 LOAD_CONST               0 (0)
             80 LOAD_CONST               3 (('args_parse',))
             82 IMPORT_NAME             11 (main)
             84 IMPORT_FROM             12 (args_parse)
             86 STORE_NAME              12 (args_parse)
             88 POP_TOP

 15          90 LOAD_CONST               0 (0)
             92 LOAD_CONST               4 (('raw_to_aug',))
             94 IMPORT_NAME             13 (aug_preprocess)
             96 IMPORT_FROM             14 (raw_to_aug)
             98 STORE_NAME              14 (raw_to_aug)
            100 POP_TOP

104         102 LOAD_CONST               5 (<code object load_raw_aug_data at 0x7ff4837feef0, file "/home/zhangjunru/robustlearn/ddlearn/data_util/raw_aug_loader.py", line 15>)
            104 LOAD_CONST               6 ('load_raw_aug_data')
            106 MAKE_FUNCTION            0
            108 STORE_NAME              15 (load_raw_aug_data)

127         110 LOAD_CONST               7 (<code object data_scaler at 0x7ff4837ff520, file "/home/zhangjunru/robustlearn/ddlearn/data_util/raw_aug_loader.py", line 104>)
            112 LOAD_CONST               8 ('data_scaler')
            114 MAKE_FUNCTION            0
            116 STORE_NAME              16 (data_scaler)

147         118 LOAD_CONST               9 (<code object reshape_data at 0x7ff4837ff730, file "/home/zhangjunru/robustlearn/ddlearn/data_util/raw_aug_loader.py", line 127>)
            120 LOAD_CONST              10 ('reshape_data')
            122 MAKE_FUNCTION            0
            124 STORE_NAME              17 (reshape_data)

179         126 LOAD_CONST              11 (<code object pick_data at 0x7ff4837ff7e0, file "/home/zhangjunru/robustlearn/ddlearn/data_util/raw_aug_loader.py", line 147>)
            128 LOAD_CONST              12 ('pick_data')
            130 MAKE_FUNCTION            0
            132 STORE_NAME              18 (pick_data)

189         134 LOAD_CONST              13 (<code object set_param at 0x7ff4837ff890, file "/home/zhangjunru/robustlearn/ddlearn/data_util/raw_aug_loader.py", line 179>)
            136 LOAD_CONST              14 ('set_param')
            138 MAKE_FUNCTION            0
            140 STORE_NAME              19 (set_param)

190         142 LOAD_NAME               20 (__name__)
            144 LOAD_CONST              15 ('__main__')
            146 COMPARE_OP               2 (==)
            148 EXTENDED_ARG             1
            150 POP_JUMP_IF_FALSE      472 (to 944)

191         152 LOAD_NAME               12 (args_parse)
            154 CALL_FUNCTION            0
            156 STORE_NAME              21 (args)

192         158 LOAD_CONST              16 ('/media/newdisk/zhangjunru/DG_Dateset/')
            160 STORE_NAME              22 (root_path)

193         162 LOAD_CONST              17 (('uschad',))
            164 GET_ITER
            166 EXTENDED_ARG             1
            168 FOR_ITER               302 (to 774)
            170 LOAD_NAME               21 (args)
            172 STORE_ATTR              23 (dataset)

194         174 LOAD_NAME               19 (set_param)
            176 LOAD_NAME               21 (args)
            178 LOAD_ATTR               23 (dataset)
            180 CALL_FUNCTION            1
            182 STORE_NAME              24 (n_domain)

195         184 LOAD_CONST              18 (('norm',))
            186 GET_ITER
            188 EXTENDED_ARG             1
            190 FOR_ITER               278 (to 748)
            192 LOAD_NAME               21 (args)
            194 STORE_ATTR              25 (scaler_method)

196         196 LOAD_CONST              19 ((0.2, 0.4, 0.6, 0.8, 1.0))
            198 GET_ITER
            200 EXTENDED_ARG             1
            202 FOR_ITER               264 (to 732)
            204 STORE_NAME              26 (remain_data_rate)

197         206 LOAD_NAME               27 (range)
            208 LOAD_CONST              20 (2)
            210 LOAD_CONST              21 (3)
            212 LOAD_CONST              22 (1)
            214 CALL_FUNCTION            3
            216 GET_ITER
            218 FOR_ITER               246 (to 712)
            220 LOAD_NAME               21 (args)
            222 STORE_ATTR              28 (seed)

198         224 LOAD_NAME               10 (utils)
            226 LOAD_METHOD             29 (set_random_seed)
            228 LOAD_NAME               21 (args)
            230 LOAD_ATTR               28 (seed)
            232 CALL_METHOD              1
            234 POP_TOP

199         236 LOAD_NAME               14 (raw_to_aug)
            238 LOAD_NAME               21 (args)
            240 LOAD_ATTR               28 (seed)
            242 LOAD_NAME               22 (root_path)
            244 LOAD_CONST               1 (None)

198         246 LOAD_NAME               21 (args)
            248 LOAD_ATTR               23 (dataset)
            250 LOAD_CONST              23 (7)
            252 LOAD_NAME               26 (remain_data_rate)

200         254 LOAD_CONST              24 (('save_path', 'dataset', 'aug_num', 'remain_data_rate'))
            256 CALL_FUNCTION_KW         6
            258 UNPACK_SEQUENCE          2
            260 STORE_NAME              30 (raw_data)
            262 STORE_NAME              31 (aug_data)

201         264 LOAD_NAME               27 (range)
            266 LOAD_NAME               24 (n_domain)
            268 CALL_FUNCTION            1
            270 GET_ITER
            272 FOR_ITER               190 (to 654)
            274 STORE_NAME              32 (target)

202         276 LOAD_NAME               22 (root_path)

201         278 LOAD_NAME               21 (args)
            280 LOAD_ATTR               23 (dataset)
            282 FORMAT_VALUE             0
            284 LOAD_CONST              25 ('/')
            286 LOAD_NAME               21 (args)
            288 LOAD_ATTR               23 (dataset)
            290 FORMAT_VALUE             0
            292 LOAD_CONST              26 ('_crosssubject_rawaug_rate')
            294 LOAD_NAME               26 (remain_data_rate)
            296 FORMAT_VALUE             0
            298 LOAD_CONST              27 ('_t')
            300 LOAD_NAME               32 (target)
            302 FORMAT_VALUE             0
            304 LOAD_CONST              28 ('_seed')
            306 LOAD_NAME               21 (args)
            308 LOAD_ATTR               28 (seed)
            310 FORMAT_VALUE             0
            312 LOAD_CONST              29 ('_scaler')
            314 LOAD_NAME               21 (args)
            316 LOAD_ATTR               25 (scaler_method)
            318 FORMAT_VALUE             0
            320 LOAD_CONST              30 ('.pkl')
            322 BUILD_STRING            12

203         324 BINARY_ADD
            326 STORE_NAME              33 (save_path)

204         328 LOAD_NAME               34 (print)
            330 LOAD_NAME               33 (save_path)
        >>  332 CALL_FUNCTION            1
            334 POP_TOP

205         336 LOAD_NAME               15 (load_raw_aug_data)

204         338 LOAD_NAME               30 (raw_data)
            340 LOAD_NAME               31 (aug_data)
            342 LOAD_NAME               21 (args)
            344 LOAD_ATTR               25 (scaler_method)
            346 LOAD_NAME               21 (args)
            348 LOAD_ATTR               23 (dataset)
            350 LOAD_NAME               32 (target)
            352 LOAD_NAME               24 (n_domain)

207         354 CALL_FUNCTION            6
            356 UNPACK_SEQUENCE         10
            358 STORE_NAME              35 (raw_trs)
            360 STORE_NAME              36 (aug_trs)
            362 STORE_NAME              37 (raw_vas)
            364 STORE_NAME              38 (aug_vas)
            366 STORE_NAME              39 (raw_trt)
            368 STORE_NAME              40 (aug_trt)
            370 STORE_NAME              41 (raw_vat)
            372 STORE_NAME              42 (aug_vat)
            374 STORE_NAME              43 (raw_tet)
        >>  376 STORE_NAME              44 (aug_tet)

208         378 LOAD_NAME               35 (raw_trs)

209         380 LOAD_NAME               36 (aug_trs)

210         382 LOAD_NAME               37 (raw_vas)

211         384 LOAD_NAME               38 (aug_vas)

212         386 LOAD_NAME               39 (raw_trt)

213         388 LOAD_NAME               40 (aug_trt)

214         390 LOAD_NAME               41 (raw_vat)

215         392 LOAD_NAME               42 (aug_vat)

216         394 LOAD_NAME               43 (raw_tet)

206         396 LOAD_NAME               44 (aug_tet)

218         398 LOAD_CONST              31 (('raw_trs', 'aug_trs', 'raw_vas', 'aug_vas', 'raw_trt', 'aug_trt', 'raw_vat', 'aug_vat', 'raw_tet', 'aug_tet'))
        >>  400 BUILD_CONST_KEY_MAP     10
            402 STORE_NAME              45 (raw_and_aug)

219         404 LOAD_NAME               46 (open)
            406 LOAD_NAME               33 (save_path)
            408 LOAD_CONST              32 ('wb')
            410 CALL_FUNCTION            2
            412 SETUP_WITH              28 (to 470)
            414 STORE_NAME              47 (f)
            416 LOAD_NAME                7 (pickle)
            418 LOAD_METHOD             48 (dump)
            420 LOAD_NAME               45 (raw_and_aug)
            422 LOAD_NAME               47 (f)
            424 CALL_METHOD              2
            426 POP_TOP
            428 POP_BLOCK
            430 LOAD_CONST               1 (None)
            432 DUP_TOP
            434 DUP_TOP
        >>  436 CALL_FUNCTION            3
            438 POP_TOP
            440 JUMP_FORWARD            18 (to 478)
            442 WITH_EXCEPT_START
            444 EXTENDED_ARG             1
            446 POP_JUMP_IF_TRUE       450 (to 900)
            448 <48>
            450 POP_TOP
            452 POP_TOP
            454 POP_TOP
            456 POP_EXCEPT
            458 POP_TOP
            460 EXTENDED_ARG             1
            462 JUMP_ABSOLUTE          272 (to 544)
            464 JUMP_ABSOLUTE          218 (to 436)
            466 JUMP_ABSOLUTE          200 (to 400)
            468 JUMP_ABSOLUTE          188 (to 376)
        >>  470 JUMP_ABSOLUTE          166 (to 332)
            472 LOAD_CONST               1 (None)
            474 RETURN_VALUE
