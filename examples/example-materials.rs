use bevy::{
    feathers::{
        FeathersPlugins,
        controls::{
            ColorChannel, ColorSlider, ColorSliderProps, SliderBaseColor, SliderProps,
            color_slider, color_swatch, slider,
        },
        dark_theme::create_dark_theme,
        theme::{ThemeBackgroundColor, UiTheme},
        tokens,
    },
    input_focus::tab_navigation::TabGroup,
    prelude::*,
    ui_widgets::{
        SliderPrecision, SliderStep, SliderValue, ValueChange, observe, slider_self_update,
    },
};

#[derive(Component)]
struct HslSwatch;

#[derive(Resource)]
struct HslWidgetStates {
    hsl_color: Hsla,
}

#[derive(Component)]
struct ActiveEntity;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(FeathersPlugins)
        .insert_resource(GlobalAmbientLight {
            brightness: 10000.0,
            color: Color::WHITE,
            ..default()
        })
        .insert_resource(UiTheme(create_dark_theme()))
        .insert_resource(HslWidgetStates {
            hsl_color: Srgba::new(0.57254905, 0.2509804, 0.05490196, 0.5).into(),
        })
        .add_systems(Startup, (setup_cube, setup_ui))
        .add_systems(Update, update_colors)
        .add_systems(Update, update_materials)
        .run();
}

fn setup_cube(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(3.0, 2.0, 3.0).looking_at(Vec3::new(0.0, -0.5, 0.0), Vec3::Y),
    ));

    let cube = meshes.add(Cuboid::new(0.5, 0.5, 0.5));
    commands.spawn((
        Mesh3d(cube.clone()),
        MeshMaterial3d(materials.add(Color::from(Hsla::new(300.0, 1.0, 0.5, 0.5)))),
        Transform::from_translation(Vec3::new(0.0, 0.0, 0.0)),
        ActiveEntity,
    ));
}

fn setup_ui(mut commands: Commands) {
    commands.spawn((
        Node {
            width: percent(20),
            height: percent(30),
            align_items: AlignItems::Start,
            justify_content: JustifyContent::Start,
            display: Display::Flex,
            flex_direction: FlexDirection::Column,
            row_gap: px(10),
            ..default()
        },
        TabGroup::default(),
        ThemeBackgroundColor(tokens::WINDOW_BG),
        children![(
            Node {
                display: Display::Flex,
                flex_direction: FlexDirection::Column,
                align_items: AlignItems::Stretch,
                justify_content: JustifyContent::Start,
                padding: UiRect::all(px(8)),
                row_gap: px(8),
                width: percent(30),
                min_width: px(200),
                ..default()
            },
            children![
                (
                    Node {
                        display: Display::Flex,
                        flex_direction: FlexDirection::Row,
                        justify_content: JustifyContent::SpaceBetween,
                        ..default()
                    },
                    children![Text("Hsl".to_owned()), color_swatch(HslSwatch),]
                ),
                (
                    color_slider(
                        ColorSliderProps {
                            value: 0.5,
                            channel: ColorChannel::HslHue
                        },
                        ()
                    ),
                    observe(
                        |change: On<ValueChange<f32>>, mut color: ResMut<HslWidgetStates>| {
                            color.hsl_color.hue = change.value
                        }
                    )
                ),
                (
                    color_slider(
                        ColorSliderProps {
                            value: 0.5,
                            channel: ColorChannel::HslSaturation
                        },
                        ()
                    ),
                    observe(
                        |change: On<ValueChange<f32>>, mut color: ResMut<HslWidgetStates>| {
                            color.hsl_color.saturation = change.value
                        }
                    )
                ),
                (
                    color_slider(
                        ColorSliderProps {
                            value: 0.5,
                            channel: ColorChannel::HslLightness
                        },
                        ()
                    ),
                    observe(
                        |change: On<ValueChange<f32>>, mut color: ResMut<HslWidgetStates>| {
                            color.hsl_color.lightness = change.value
                        }
                    )
                ),
                (
                    color_slider(
                        ColorSliderProps {
                            value: 0.5,
                            channel: ColorChannel::Alpha
                        },
                        ()
                    ),
                    observe(
                        |change: On<ValueChange<f32>>, mut color: ResMut<HslWidgetStates>| {
                            color.hsl_color.alpha = change.value
                        }
                    )
                ),
                (
                    slider(
                        SliderProps {
                            max: 100.0,
                            value: 20.0,
                            ..default()
                        },
                        (SliderStep(1.0), SliderPrecision(2),)
                    ),
                    observe(slider_self_update)
                ),
            ]
        ),],
    ));
}

fn update_materials(
    material_handles: Query<&MeshMaterial3d<StandardMaterial>, With<ActiveEntity>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    color: Res<HslWidgetStates>,
) {
    for material_handle in material_handles.iter() {
        if let Some(material) = materials.get_mut(material_handle)
            && let Color::Hsla(ref mut hsla) = material.base_color
        {
            *hsla = color.hsl_color;
            // info!("hsla alpha is {}", hsla.alpha);
        }
    }
}

fn update_colors(
    color: Res<HslWidgetStates>,
    mut sliders: Query<(Entity, &ColorSlider, &mut SliderBaseColor)>,
    swatches: Query<&Children, With<HslSwatch>>,
    mut commands: Commands,
) {
    if color.is_changed() {
        for (slider_ent, slider, mut base) in sliders.iter_mut() {
            match slider.channel {
                ColorChannel::HslHue => {
                    base.0 = color.hsl_color.into();
                    commands
                        .entity(slider_ent)
                        .insert(SliderValue(color.hsl_color.hue));
                }

                ColorChannel::HslSaturation => {
                    base.0 = color.hsl_color.into();
                    commands
                        .entity(slider_ent)
                        .insert(SliderValue(color.hsl_color.saturation));
                }

                ColorChannel::HslLightness => {
                    base.0 = color.hsl_color.into();
                    commands
                        .entity(slider_ent)
                        .insert(SliderValue(color.hsl_color.lightness));
                }
                ColorChannel::Alpha => {
                    base.0 = color.hsl_color.into();
                    commands
                        .entity(slider_ent)
                        .insert(SliderValue(color.hsl_color.alpha));
                }

                _ => {}
            }
        }

        for children in swatches.iter() {
            commands
                .entity(children[0])
                .insert(BackgroundColor(color.hsl_color.into()));
        }
    }
}
