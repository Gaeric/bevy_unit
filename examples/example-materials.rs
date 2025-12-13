use bevy::{
    color::palettes::tailwind::AMBER_800,
    feathers::{
        FeathersPlugins,
        controls::{
            ColorChannel, ColorSlider, ColorSliderProps, SliderBaseColor, color_slider,
            color_swatch,
        },
        dark_theme::create_dark_theme,
        theme::{ThemeBackgroundColor, UiTheme},
        tokens,
    },
    input_focus::tab_navigation::TabGroup,
    prelude::*,
};

#[derive(Component)]
struct HslSwatch;

#[derive(Resource)]
struct HslWidgetStates {
    hsl_color: Hsla,
}
fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(FeathersPlugins)
        .insert_resource(UiTheme(create_dark_theme()))
        .insert_resource(HslWidgetStates {
            hsl_color: AMBER_800.into(),
        })
        .add_systems(Startup, setup)
        .add_systems(Update, update_colors)
        .run();
}

fn setup(
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
        MeshMaterial3d(materials.add(Color::from(Hsla::hsl(300.0, 1.0, 0.5)))),
        Transform::from_translation(Vec3::new(0.0, 0.0, 0.0)),
    ));

    commands.spawn((
        Node {
            width: percent(100),
            height: percent(100),
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
                (color_slider(
                    ColorSliderProps {
                        value: 0.5,
                        channel: ColorChannel::HslHue
                    },
                    ()
                ),),
                (color_slider(
                    ColorSliderProps {
                        value: 0.5,
                        channel: ColorChannel::HslSaturation
                    },
                    ()
                )),
                (color_slider(
                    ColorSliderProps {
                        value: 0.5,
                        channel: ColorChannel::HslLightness
                    },
                    ()
                ))
            ]
        ),],
    ));
}

fn update_materials(
    material_handles: Query<&MeshMaterial3d<StandardMaterial>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    for material_handle in material_handles.iter() {
        if let Some(material) = materials.get_mut(material_handle)
            && let Color::Hsla(ref mut hsla) = material.base_color
        {
            *hsla = hsla.rotate_hue(1.0 / 60.0 * 100.0);
        }
    }
}

fn update_colors(
    colors: Res<HslWidgetStates>,
    mut sliders: Query<(Entity, &ColorSlider, &mut SliderBaseColor)>,
    swatches: Query<(&HslSwatch, &Children)>,
    mut commands: Commands,
) {
    if colors.is_changed() {
        for (slider_ent, slider, mut base) in sliders.iter_mut() {
            match slider.channel {
                _ => {
                    info!("12341234241234")
                }
            }
        }
    }
}
