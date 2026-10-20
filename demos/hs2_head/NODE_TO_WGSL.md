# HS2 Blender 材质 → WGSL 对照手册

本文整理 HS2 角色材质从 Blender 节点树到 Bevy/WGSL 的完整转换链，
用于指导 `demos/hs2_head/src` 里各部位材质的还原与后续补全。

- 数据来源：Blender 插件 `hs2_blender_export`（负责在 Blender 里搭建节点树材质）
- 参考规格：`assets/materials/minified_nodes.json`（节点树快照）
- 转换目标：`assets/materials/shaders/hs2_head_*.wgsl` + `demos/hs2_head/src/*.rs`

---

## 1. 一句话结论

**这套链路里"材质 → WGSL"不是自动翻译，而是"Blender 节点树 = 参考规格书 →
人/LLM 手写 Rust 扩展材质 + WGSL"的手工转换。**
`minified_nodes.json` 只是给作者/LLM 看的中间规格；GLB 里的 glTF 材质只带名字和基础色，
真正的着色资源是在运行时代码里按文件路径重新加载的贴图 + 手写 shader。

---

## 2. 相关文件清单

| 仓库 | 文件 | 作用 |
|---|---|---|
| `hs2_blender_export` | `importer.py` | append 材质模板、`set_tex`/`set_bump` 填贴图、按像素统计设 Bump gamma |
| `hs2_blender_export` | `assets/prefab_materials_meshexporter.blend` | 材质模板（含自定义 Shader 节点组的内部实现） |
| `hs2_blender_export` | `attributes.py` | 通过节点 default_value 暴露 UI 参数（发色、眉毛 scale 等） |
| `bevy_unit` | `units/export_blender_nodes.py` | 材质节点树 → `minified_nodes.json` |
| `bevy_unit` | `assets/materials/minified_nodes.json` | 22 个材质的节点树快照 |
| `bevy_unit` | `assets/materials/hs2_*.glb` | glTF 导出（几何 + 材质名 + 基础色，无贴图） |
| `bevy_unit` | `demos/hs2_head/src/mat_convert.rs` | 按 glTF 材质名替换成 `ExtendedMaterial` |
| `bevy_unit` | `demos/hs2_head/src/{head,body,eye,eyelash,eyeshadow}.rs` | 每部位的扩展材质 / 绑定声明 |
| `bevy_unit` | `assets/materials/shaders/hs2_head_*.wgsl` | fragment shader 实现 |

---

## 3. 转换链条（5 步）

```
HS2 游戏贴图/dump
  │  hs2_blender_export：append 材质模板 + set_tex/set_bump
  ▼
Blender 材质（节点树；HS2 皮肤/眼睛逻辑在自定义 Shader 节点组内）
  │  (1) export_blender_nodes.py  →  minified_nodes.json（规格书）
  │  (2) Blender glTF 导出        →  .glb（几何 + 材质名 + 基础色）
  ▼
Bevy 加载 .glb → 每个实例带 (MeshMaterial3d<StandardMaterial>, GltfMaterialName)
  │  mat_convert.rs：按 glTF 材质名查 MaterialRegistry
  ▼
替换成 ExtendedMaterial<StandardMaterial, PartExt>
  │  head.rs/eye.rs/... 声明贴图/采样器/uniform 绑定号
  ▼
渲染时 MaterialPlugin 编译各 hs2_head_*.wgsl，fragment 里改写
StandardMaterial 的 base_color/emissive/... 再走 bevy_pbr 的 apply_pbr_lighting
```

实测验证：`hs2_body_greybox_mini_2.glb` 里所有材质 `textures: 0`，
只有 `baseColorFactor`/`roughnessFactor`。即 GLB/StandardMaterial 只负责：

1. 每个部件的材质**名字**（`Head_`/`Eyes_`/`Torso_`… → 触发哪个扩展）；
2. 给 `pbr_input_from_standard_material()` 提供基础属性（UV、法线等）。

---

## 4. 材质注册（转换的"开关"）

`mat_convert.rs` 中 `MatConvertPlugin` 只对 5 个名字启用自定义 WGSL：

```rust
(EyeMaterialExt,       "Eyes_"),
(EyelashMaterialExt,   "Eyelashes_"),
(EyeshadowMaterialExt, "Eyeshadow_"),
(HeadMaterialExt,      "Head_"),
(BodyMaterialExt,      "Torso_")
```

流程（`update_material`）：

```rust
On<WorldInstanceReady> → 遍历后代
  → 命中 GltfMaterialName → 克隆 StandardMaterial 作为 base
  → queue 里查 MaterialRegistry → applier.apply(entity, &base_mat, world)
  → 移除 MeshMaterial3d<StandardMaterial>，插入 MeshMaterial3d(extended_handle)
```

`minified_nodes.json` 中其它材质（`clothing_*`、`hair_*`、`Nails_`、`Gums`、
`Teeth.002`、`Tongue_`、`Exhaust_Armature`）落到 `DefaultTransparentApplier`，
不换 shader，保留 glTF 标准材质。

---

## 5. Rust 绑定 ↔ WGSL 绑定如何对齐

扩展材质同时支持 bindless / 非 bindless，由 WGSL 的 `#ifdef BINDLESS` 切换。

以 `Head_` 为例。Rust 侧（`head.rs`）：

```rust
#[data(80, HeadMaterialUniform, binding_array(106))]
#[bindless(index_table(range(80..93), binding(105)))]
struct HeadMaterialExt {
    #[texture(81)] #[sampler(82)] main_texture: Option<Handle<Image>>,
    #[texture(83)] #[sampler(84)] detail_texture,
    #[texture(85)] #[sampler(86)] detail_gloss_texture,
    #[texture(87)] #[sampler(88)] eyebrow_texture,
    #[texture(89)] #[sampler(90)] bump_texture,
    #[texture(91)] #[sampler(92)] bump_ex_texture,
}
```

WGSL 侧声明同一批 binding 号（`#{MATERIAL_BIND_GROUP}` 是 bevy 预处理符）：

```wgsl
struct HeadMaterialExtIndices { material, main_texture, main_sampler, ... }

@group(#{MATERIAL_BIND_GROUP}) @binding(105) var<storage> head_material_ext_indices: array<HeadMaterialExtIndices>;
@group(#{MATERIAL_BIND_GROUP}) @binding(106) var<storage> head_material_ext: array<HeadMaterialExt>;

// 非 bindless 分支：
@group(#{MATERIAL_BIND_GROUP}) @binding(81) var main_texture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(82) var main_sampler: sampler;
...
```

所有 fragment shader 的统一主流程：

```wgsl
var pbr_input = pbr_input_from_standard_material(in, is_front);
let uv = (uv_transform * vec3(in.uv, 1.0)).xy;
// ...贴图采样 + 数学运算，改写 pbr_input.material.base_color / emissive / ...
out.color = apply_pbr_lighting(pbr_input);
out.color = main_pass_post_lighting_processing(pbr_input, out.color);
```

即：Blender 的 **Principled BSDF 出口由 `apply_pbr_lighting` 接管**，
节点图里"产生颜色/属性"的部分搬进 fragment shader。

---

## 6. 逐材质：节点树 → WGSL

### 6.1 `Head_`（皮肤头部，最复杂）—— WIP，仅占位

外层节点图（`minified_nodes.json`）：

```
Texture Coordinate[UV]
   ├─ Vector Math.003 (MULTIPLY [24,12,1], Scale 5) → DetailGlossMap[Vector]  ← 高频毛孔 gloss 的 UV 平铺
Eyebrow rotation / X offset / Y offset / scale / arch (ShaderNodeValue)
   └─ Group.002 (Degrees/Scale/Green/Red/Value) → Eyebrow[Vector]             ← 眉毛 UV 旋转/缩放/平移

MainTex → Shader[Main texture]         DetailMainTex → Shader[Detail texture]
DetailGlossMap(Color, Alpha) → Shader  RGB → Shader[Eyebrow color]
Eyebrow → Shader[Eyebrow texture]      BumpMap → Shader[Bump image]   BumpMap2 → Shader[Bump image 2]

Shader[BSDF] → Material Output[Surface]
Shader[Displacement] → Material Output[Displacement]
```

"Shader" 是自定义节点组（真正的 HS2 皮肤着色器），JSON 只暴露接口输入。
接口 ↔ Rust/WGSL 对照：

| Blender Shader 组输入（JSON 默认值） | Rust `HeadMaterialExt`/`HeadMaterialUniform` | 当前 WGSL 状态 |
|---|---|---|
| `Main texture`（← MainTex） | `main_texture` | ✅ 采样 → base_color |
| `Detail texture`（← DetailMainTex） | `detail_texture` | ❌ 未消费 |
| `DetailGlossMap` / alpha | `detail_gloss_texture` | ❌ 未消费 |
| `Eyebrow texture` / color | `eyebrow_texture`、`eyebrow_color` | ❌ 未消费 |
| `Bump image`、`Bump scale`、`Bump gamma R/G` | `bump_texture` | ❌ 未消费 |
| `Bump image 2`、…（男角会置 0） | `bump_ex_texture` | ❌ 未消费 |
| `Subsurface Strength`(0.506) | `subsurface_strength` | ❌ |
| `Subsurface Color`(蓝紫调) | `subsurface_color` | ❌ |
| `Textured skin gloss`(1.0)、`delta`(0.75) | `skin_gloss` | ❌ |
| `Normal strength`(1.0) | `normal_strength` | ❌ |
| （Bevy 侧追加） | `clear_coat` | ❌ |

当前 `hs2_head_head_material.wgsl` 实际逻辑（占位）：

```wgsl
let main_color = textureSample(main_texture, main_sampler, uv);
// HSV 简单色调映射：近似游戏贴图曲线
var hsv_color = rgb_to_hsv(main_color.xyz);
hsv_color.z = sqrt(sqrt(hsv_color.z));
pbr_input.material.base_color = vec4(hsv_to_rgb(hsv_color), main_color.a);
```

> 结论：Head WGSL 只还原了 `MainTex → base_color（色调映射）` 一个叶子，
> Rust 的 6 张贴图槽位和 uniform 已按规格书预留，但 shader 尚未消费。

### 6.2 `Torso_`（身体皮肤）—— WIP，仅占位

```
Texture Coordinate → Vector Math.010 (MULTIPLY [24,12,1], Scale 10) → DetailGlossMap
Attribute[Color] → Group.003[Value] ;  Booby scale → Group.003[Value]
Texture2 → Shader[Nipple texture]
```

Shader 组输入比 Head 多：`Nipple texture/color`、`Nipple bump strength`、
`Veins`、`Subsurface radius scale`。

Rust `BodyMaterialExt` 目前只声明 `main_texture`；
`hs2_head_body_material.wgsl` 同样只有 MainTex + HSV 色调映射。**占位阶段。**

### 6.3 `Eyes_`（对照最完整）—— 结构已还原，参数/色彩 WIP

外层节点图（`minified_nodes.json`）：

```
Iris texture (Image Texture.001)
  └─ Separate RGB
       ├─ .R ───────────────→ Math.001(MULTIPLY) → Mix.004[Factor]
       ├─ .B ───────────────→ Mix.001[Factor]    ← 巩膜/虹膜切换
       └─ .G → Math.002 → Principled[Specular IOR Level]
Pupil size → Group(Scale) → Image Texture.002(瞳孔)
  └─ .Alpha → Math(SUBTRACT) → Math.001(×iris.R)
Iris size → Group.001(Scale) → 各 Image Texture 的 UV（居中缩放）
Sclera (Image Texture) → Vector Math(MULTIPLY 0.84) → Mix.001[A]
RGB(虹膜基色) → Mix.004[B]
Highlight strength → Mix.002(ADD)[Factor] 高光贴图
Mix.003 → Emission Color ； Value → Emission Strength
```

`hs2_head_eye_material.wgsl` 几乎逐行对应：

```wgsl
// ShaderNodeVectorMath MULTIPLY 0.84
let multiply1_result = sclera_color * vec4<f32>(0.84);

// ShaderNodeMath SUBTRACT：1 - Rec.709 亮度(瞳孔)
let substract_value_b = dot(pupil_color.rgb * pupil_color.a, rec_709_coeffs);
let multiply_result = substract_value_a * substract_value_b * iris_color.r; // ×iris.R

let mix1_result = mix(vec4(0), iris_base_color, saturate(multiply_result)); // Mix.004
let mix_result  = mix(multiply1_result, mix1_result, saturate(iris_color.b)); // Mix.001
let add_result  = color_a + saturate(add_factor) * highlight_color;          // Mix.002 ADD
```

已知 TODO（WGSL 注释）：`pupil_uv = (uv - 0.5) * 3.7 + 0.5` 硬编码，
`Iris size`/`Pupil size`/`Highlight strength` 尚未进 uniform；色彩空间存疑
（注释 `// WA now, can't understand color space`）。

### 6.4 `Eyelashes_`（简单 alpha 材质）—— 基本完成

节点图：`Principled BSDF` + `Image Texture → Separate RGB → .R 接 Alpha`
（DXT 纹理 R 通道当 alpha 遮罩）+ `RGB → Base Color`（发色）。

WGSL：

```wgsl
pbr_input.material.base_color = vec4<f32>(0.0, eyelash_color.g, eyelash_color.b, eyelash_color.r);
```

Rust：`alpha_mode = AlphaMode::Blend`，贴图以 `is_srgb = true` 加载
（`c_t_eyelash_04-DXT1.dds`）。

### 6.5 `Eyeshadow_`（简单 alpha 材质）—— 基本完成

节点图：`Principled BSDF` 底色固定深色 `0.016`，`Alpha` 接 `Image Texture[Alpha]`；
另有一个未使用的 `Transparent BSDF → Material Output.001`。

WGSL：

```wgsl
pbr_input.material.base_color = vec4<f32>(0.016, 0.016, 0.016, eyeshadow_color.a);
```

Rust：`alpha_mode = AlphaMode::Blend`（`c_t_eyeshadow_02-DXT5.dds`）。

---

## 7. 通用转换规律

1. Blender **常量颜色 / Value 节点** → Rust struct 字段 → WGSL uniform / storage。
2. **贴图节点** → `#[texture(n)] #[sampler(n)]` → 运行时 `AssetServer::load_builder()`
   （含 `is_srgb`、ClampToBorder 等加载参数）→ `textureSample`。
3. **UV 操作组**（`Group`/`Group.001` 居中缩放）→ 用 glTF `uv_transform` 或
   `(uv - 0.5) * k + 0.5`。
4. **Math / Mix / Vector Math 节点** → 直接 WGSL 标量/向量运算
   （`mix`、`*`、`dot`、`1 - x`）。
5. **Principled BSDF → 结果** → 改写 `pbr_input.material.*`，交给 `bevy_pbr` 光照。
6. 转换**没有自动化工具**；`export_blender_nodes.py` 故意只输出 minified JSON 供 LLM/人阅读。

---

## 8. 用导出数据做"逐节点解析"的固有限制

- **导出器不递归 group 内部**：HS2 皮肤逻辑在 `Shader`（Head/Torso）与
  `Group`/`Group.001`/`Group.002` 的组内节点树里，JSON 只留接口输入与默认值。
- **不记录 Image Texture 的文件名/色彩空间**，需从 `hs2_blender_export/importer.py`
  的 `set_tex(..., csp=...)` 与各 `*.rs` 的路径反查。
- **同名 socket 覆盖**：Math 的两个 "Value"、Mix 的 Factor 等同名输入会互相覆盖，
  个别连接会丢。
- **字符串被序列化成字符数组**：`"operation": ["M","U","L",...]`，需 `"".join()` 还原。
- glTF 导出**丢弃全部节点树**，只留 `baseColorFactor`/`roughnessFactor`。

要补全 Head/Torso 的完整逐节点映射，需要打开
`hs2_blender_export/assets/prefab_materials_meshexporter.blend` 中的 `Head`/`Torso`
材质模板及其 Shader 节点组内部结构。

---

## 9. 进度清单（TODO）

- [ ] `Head_`：detail texture（DetailMainTex/DetailGlossMap 平铺采样）
- [ ] `Head_`：eyebrow UV transform + eyebrow 混合
- [ ] `Head_` / `Torso_`：双层 bump（gamma R/G 处理）→ normal
- [ ] `Head_`：subsurface（strength/color/mix）、clear coat、skin gloss delta
- [ ] `Torso_`：nipple 链（Attribute/Booby scale → Texture2 UV）
- [ ] `Eyes_`：`Iris size`/`Pupil size`/`Highlight strength` 进 uniform（替换硬编码 3.7）
- [ ] `Eyes_`：色彩空间核对（linear/sRGB、DXT1 解码）
- [ ] 清理 `export_blender_nodes.py` 的序列化 bug（字符串变字符数组、typo `opration`）


## 10. 贴图转换规则（人类可读版）

这一节只讲"一张素材从磁盘到 shader 里一个采样值"之间发生了什么，
不涉及节点图本身。

### 10.1 素材形态

| 来源 | 形态 | 说明 |
|---|---|---|
| HS2 dump | DDS，多为 DXT1 / DXT5 | DXT1 无 alpha 或 1-bit alpha；DXT5 带插值 alpha。加载后统一成 RGBA8 |
| MeshExporter | PNG | 从 dump 重新编码，并额外生成 `*_converted` |
| `_converted` | PNG | 把游戏里的高度/凹凸图转成**切线空间法线图**（B≈1，R/G 有变化） |

### 10.2 色彩空间

规则只有一条：`importer.py` 里 `set_tex(..., csp='Non-Color')` 的一定按线性读，
`csp=None` 的按 sRGB 读。

| 槽位 | 色彩空间 | 备注 |
|---|---|---|
| `MainTex`（皮肤/头发） | sRGB | 同时 `alpha_mode='NONE'`，alpha 不参与混合 |
| `DetailMainTex` | 线性 | |
| `DetailGlossMap` | 线性 | 只读 alpha（高频 gloss 遮罩） |
| `Texture3`（眉毛） | 线性 | Cubic 插值 |
| `BumpMap*_converted` | 线性 | 见 10.3 |
| `SubsurfaceAlbedo` | 线性 | 本地缺失 |
| 眼睛 `MainTex`(sclera) / `Texture2`(iris) | sRGB | |
| 眼睛 `Texture3`(pupil) / `Texture4`(highlight) | 线性 | 瞳孔 `CLIP` 寻址 |
| 睫毛 `MainTex` | 线性 | R 通道当 alpha 遮罩 |
| 眼影 `MainTex` | sRGB | alpha 混合 |

### 10.3 法线贴图：为什么需要逐通道 gamma

`_converted` 已经是法线图，但 R/G 的**中位数并不在 0.5**。实测：

| 文件 | R 中位数 | G 中位数 |
|---|---|---|
| `cf_m_skin_head_01_BumpMap_converted.png` | 0.498 | 0.647 |
| `cf_m_skin_body_00_BumpMap_converted.png` | 0.498 | 0.647 |

如果直接按 `2c-1` 解码，所有法线会带一个恒定的 +Y 倾斜。`importer.py`
用中位数反推 gamma，让中位数回到 0.5：

```
gamma = -1 / log2(median)        // median^gamma = 0.5
```

代入实测值得到 **R: 1.0，G: 1.59~1.62**，这正是 prefab 里 `Bump gamma R/G`
默认值的来源。所以转换到 WGSL 时：

```
n.xy = 2 * (pow(c.xy, gamma) - 0.5)
n.z  = 1
n    = normalize(n)
```

B 通道丢弃（本来就恒为 1）。两张法线叠加的顺序是：各自解码 → xy 相加 →
归一化；这与 Blender 的 `Bump Adjust → ADD → Normal Map` 等价。

如果只拿到 `BumpMap.png`（高度图，R=B=255、G 有变化）：先用
`test_inverted_bump_map` 判断它是否被误标成 `_converted`，再按同样方式估 gamma。

### 10.4 平铺与寻址

| 槽位 | Blender 模式 | 等价加载设置 |
|---|---|---|
| `MainTex` / `DetailMainTex` | `REPEAT` | `Repeat` |
| `DetailGlossMap` | `REPEAT` + UV 缩放 | `Repeat`；Head `(120, 60)`，Torso `(240, 120)`（即 24×12 再乘 5 / 10） |
| 眉毛 `Texture3` | `EXTEND` | `ClampToEdge` |
| 法线图 | `EXTEND` | `ClampToEdge` |
| 瞳孔 `Texture3` | `CLIP` | `ClampToBorder` + 透明黑边框 |

UV 缩放来自节点图里的 `Vector Math`（`MULTIPLY (24, 12, 1)` 再 `SCALE 5/10`），
不是贴图自身属性，所以必须写进 shader 参数，而不是加载设置。

### 10.5 眉毛 UV 链

```
uv = max(uv, 1 - uv)                                  // X 镜像，一张图覆盖左右眉
uv = rotate(uv, center=(0.65, 0.6), radians(9°))
uv = uv * 4
uv += (X_offset, Y_offset + (uv.x - 0.6)² * arch)
uv -= (2, 2)
```

镜像这一步（`NodeGroup.008` 的 `MULTIPLY_ADD → MAXIMUM`）是之前文档漏掉的；
它决定了一张眉毛图如何同时贴到两侧。末尾的 `-2` 在默认参数下会产生负 UV，
由 `EXTEND` clamp；保留与否用参考渲染比对后再定。

### 10.6 缺失素材的占位策略

| 占位 | 像素值 | 用于 | 效果 |
|---|---|---|---|
| White | (255,255,255,255) | MainTex / DetailMainTex | 白色底，不引入暗色 |
| FlatNormal | (128,128,255,255) | 法线图 | 解码后是 (0,0,1)，不扰动法线 |
| GlossNeutral | (255,255,255,255) | DetailGlossMap | alpha=1，gloss 走中性值 |
| TransparentBlack | (0,0,0,0) | Subsurface 等可选层 | 相当于该层不存在 |

这样 hair / clothing 这类素材还没到的材质也能出正确形状的结果，而不是黑块或全透明。

### 10.7 落地位置

- 清单：`demos/hs2_head/src/bake/textures.rs` 的 `TextureSlot`
  （`path / is_srgb / address / placeholder`）。
- 加载：`TextureSlot::load()` 统一走 `load_builder().with_settings(...)`，
  缺失时返回 1×1 占位纹理。
- 参数：UV 平铺、gamma、眉毛变换全部进 bake uniform，不写死在 shader 里。


## 11. 当前的 bake 实现

`§6.1` / `§6.2` 里描述的 fragment-shader 占位实现已被本节取代，节点语义仍然有效。

### 11.1 数据流

```
glTF 材质名 (Head_ / Torso_ / Eyes_ / Eyelashes_ / Eyeshadow_ / hair_)
  → MaterialRegistry → BakeApplier
  → compute pass (8x8) → 3 张输出纹理
  → StandardMaterial { base_color_texture, normal_map_texture, metallic_roughness_texture }
```

每个 recipe 一次 dispatch，输出与 bake 分辨率一致的 UV 空间贴图。
base color 用 `Rgba16Float`（线性，避免 8-bit 暗部断层），法线与
metallic-roughness 用 `Rgba8Unorm`（数据贴图，不能走 sRGB 解码）。

### 11.2 绑定布局

| 材质 | 输入 | 输出 |
|---|---|---|
| `Head_` / `Torso_`（共用 `hs2_head_bake_skin.wesl`） | 0-11：main / detail / detail_gloss / eyebrow / bump / bump2（各含 sampler） | 13 base(16F)、14 normal、15 mr |
| `hair_*`（`hs2_head_bake_hair.wesl`） | 0-3：main / bump | 5 base、6 normal、7 mr |
| `Eyes_` / `Eyelashes_` / `Eyeshadow_` | 各自原有输入，输出统一改成 `Rgba16Float` | base |

skin shader 的 uniform 在 binding 12。

### 11.3 参数来源

| 参数 | Head | Torso |
|---|---|---|
| `uv_transform` | (1, 1, 0, 0) | (1, 1, 0, 0) |
| `detail_uv` | (120, 60) | (240, 120) |
| `eyebrow` | 9°, ×4, arch 0, 末尾 -2 | 不用 |
| `eyebrow_color` | (0.0657, 0.3280, 0.0) | 不用 |
| `gloss` | textured 1.0 / delta 0.75 / base 0 / coat 0 | 同左 |
| `bump_gamma` | (1.0, 1.592) ×2 | 同左 |
| flags | eyebrow + mirror + coat_from_detail | 0 |

### 11.4 由节点图得到的算式

```
color   = mix(main, subsurface_color, subsurface_mix)
color   = mix(color, eyebrow_color, eyebrow.b)                 # Head
gloss   = base_gloss + textured_gloss*(1-eb.b) * 2*(dgm.a - delta)
rough   = clamp((0.5 - 0.3*gloss) * (1 - coat), 0.04, 1)
normal  = normalize(vec3(2*(c1.xy^g1 - 0.5)*s1 + 2*(c2.xy^g2 - 0.5)*s2, 1))
hair    = (base_color = hair_color, alpha = tex.a, metallic = tex.g, rough = 0.4364)
```

### 11.5 仍未移植（有意为之）

| 项 | 原因 |
|---|---|
| `Skin effect`（Voronoi/Noise 毛孔） | Blender 插件增强，非游戏材质 |
| `Bump` 高度节点 | 需要屏幕空间导数，纹理空间 bake 无法精确表达 |
| Wave Texture 头发置换 | 渲染期置换，非材质属性 |
| 次表面散射 | `StandardMaterial` 没有 SSS；只保留亮度混合 |
| `Specular IOR Level = gloss` | 需要额外一张贴图承载逐像素值，暂未做 |

### 11.6 相关文件

- `src/bake/textures.rs`：贴图清单与占位符
- `src/bake/head.rs` / `body.rs`：skin 参数
- `src/bake/hair.rs`：头发
- `assets/materials/shaders/hs2_head_bake_skin.wesl` / `_hair.wesl`：compute shader


## 执行要求
1. 不得主动调起任何可能会打开窗口的界面，只有能以headless方式执行的命令才被允许执行
2. 现在要求全部移植过程走bake方法，而不走原先的ext material
3. 目前只需要考虑材质，先把所有材质都正确移植过去
4. 全程写英文注释，不要写中文注释
5. 把对贴图的转换方法用人类可读的语言总结到文档


