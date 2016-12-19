require 'test_helper'

class SessionControllerTest < ActionController::TestCase
  test "get index" do
    get :index
    assert_response :success
  end

  test "log in" do
    post :create, params: { authorisation_token: 'test' }
    assert_redirected_to data_import_path

    post :create, params: { authorisation_token: 'test1' }
    assert_redirected_to root_path
  end

end