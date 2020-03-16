from estimate_homography import *


def run_point_est(world_pts, img_pts, img_path, savefig_prefix=""):

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)



    if isinstance(img_pts, list):
        img_pts = np.array(img_pts)

    if isinstance(world_pts, list):
        world_pts = np.array(world_pts)

    plt.figure()
    plt.imshow(img)
    plt.scatter(img_pts[:,0], img_pts[:,1], color='red')
    plt.axis("off")
    plt.title("Original image with img points marked in red")
    plt.savefig(savefig_prefix+"_1.jpg")


    H = calculate_homography(img_pts, world_pts)  # img_pts = H * world_pts

    #### cross check ####

    t_one = np.ones((img_pts.shape[0],1))
    t_out_pts = np.concatenate((world_pts, t_one), axis=1)
    x = np.matmul(H, t_out_pts.T)
    x = x/x[-1,:]

    print(" img_pts given: {}".format(img_pts))

    print("img_pts_calculated: {}".format(x.T))

    print("homography matrix estimated by opencv: {}".format(cv2.findHomography(world_pts, img_pts)[0]))

    print("homography matrix calculated: {}".format(H))

    #####################
    # without offset correction.

    world_img = np.zeros_like(img)


    mask = np.ones((img.shape[0], img.shape[1]))
    out = fit_image_in_target_space(img, world_img, mask, H)
    plt.figure()
    plt.imshow(out)
    plt.title("Corrected image with Point point correspondence - NO OFFSET CORRECTION")
    plt.axis("off")
    plt.savefig(savefig_prefix + "_2.jpg")

    #####################
    # with offset correction.

    h, w, _ = img.shape

    # Figure out where the corners of image map to the world coordinates
    corners_img = np.array([[0, 0], [w, 0], [w, h], [0, h]])
    H_inv = np.linalg.inv(H)

    t_out_pts = np.concatenate((corners_img, t_one), axis=1)
    world_crd_corners = np.matmul(H_inv, t_out_pts.T)
    world_crd_corners = world_crd_corners/world_crd_corners[-1, :]  # cols of [x1, y1, z1]

    min_crd = np.amin(world_crd_corners.T, axis=0)
    max_crd = np.max(world_crd_corners.T, axis=0)

    offset = min_crd.astype(np.int64)
    offset[2] = 0  # [x_offset, y_offset, 0]

    width_world = np.ceil(max_crd - min_crd)[0] + 1
    height_world = np.ceil(max_crd - min_crd)[1] + 1

    world_img = np.zeros((int(height_world), int(width_world), 3), dtype=np.int64)
    mask = np.ones((int(height_world), int(width_world)))

    out = fit_image_in_target_space(img, world_img, mask, H, offset)

    plt.figure()
    plt.imshow(out)
    plt.axis("off")
    plt.title("Corrected image with Point point correspondence")
    plt.savefig(savefig_prefix + "_3.jpg")
    plt.show()


if __name__ == "__main__":

    img_path = "/Users/aartighatkesar/Documents/homography_distortion_removal/Original_Images/1.jpg"

    # [[tl- top left], [tr- top right], [br- bottom right], [bl- bottom left]]
    # given, the height and width of the planar objects found in the image
    world_pts = [[0, 0], [60, 0], [60, 80], [0, 80]]

    img_pts = [[1260, 760], [1383, 705], [1372, 891], [1245, 942]]  # 5th window from right

    run_point_est(world_pts, img_pts, img_path, savefig_prefix="p1")

    ##############################

    img_path = "/Users/aartighatkesar/Documents/homography_distortion_removal/Original_Images/2.jpg"

    # [[tl- top left], [tr- top right], [br- bottom right], [bl- bottom left]]
    # given, the height and width of the planar objects found in the image
    world_pts = [[0, 0], [40, 0], [40, 80], [0, 80]]

    img_pts = [[246, 71], [327, 82], [323, 265], [245, 269]]  # 5th window from right

    run_point_est(world_pts, img_pts, img_path, savefig_prefix="p2")


